# -*- coding: utf-8 -*-
"""
Image + text Qwen2.5-VL: default TRL GRPO.

- Default TRL GRPO per-token loss: exp(logp - logp.detach()) * A + beta * KL.
- Optional --use_latent_noise_ratio: for <|latent_pad|>, replace the surrogate term with
  r = exp(-0.5 * cos(h_now, h_noise.detach())); h_noise is built after rollout by adding
  Gaussian noise to the completion last-hidden state at latent positions;
  optional --latent_noise_rollout_logit_std>0 adds noise to that token's logit during generate
  so different rollouts sample different latent tokens.
- Optional --use_latent_traj_sim_reward: using correct and incorrect rollouts in a group, define
  sim(tau_i, tau_j) as the mean cosine similarity of last-hidden trajectories at <|latent_pad|>
  positions, and replace the scalar reward with a contrastive log-softmax against target/neg
  (--latent_traj_sim_temp is tau).
  On multiple GPUs the G rollouts are split across processes. After accelerate.gather the order
  matches TRL GRPO; similarity is computed on the gathered tensors, then sliced back to this rank.
- Multimodal adapters: _encode_vl_prompt, _get_per_token_logps, _prepare_inputs (optional logit preprocessing).

Run (from src):
  python grpo_latent_easy.py --load_model_path /path/ckpt --data_path ../data/.../grpo_direct_3456.jsonl --save_model_path /path/out

Multi-GPU:
  accelerate launch --config_file config/default_config.yaml src/grpo_latent_easy.py --accelerate ... same arguments as above
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import re
from functools import partial
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import gather, gather_object
from datasets import Dataset, Features, Value
from datasets.features import Json
from PIL import Image as PILImage
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    LogitsProcessor,
    LogitsProcessorList,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    TrainerCallback,
    Trainer as HfTrainer,
)
from trl import GRPOConfig, GRPOTrainer
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import selective_log_softmax

from task import simulate_vsp, task_preporcess_config
from utils import load_jsonl_dataset, place_input_image, seed_everything


class LatentPadLogitNoiseProcessor(LogitsProcessor):
    """
    During one decoding step, add Gaussian noise to the logit at the vocab index of <|latent_pad|>
    so each rollout samples a different latent token.
    scores: [B, vocab]
    """

    def __init__(self, latent_pad_id: int, std: float, generator: Optional[torch.Generator] = None):
        if latent_pad_id < 0:
            raise ValueError("latent_pad_id 无效")
        self.latent_pad_id = int(latent_pad_id)
        self.std = float(std)
        self.generator = generator

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.std <= 0.0:
            return scores
        n = int(scores.shape[0])
        g = self.generator
        for i in range(n):
            if g is not None:
                nvec = torch.randn((), device=scores.device, dtype=scores.dtype, generator=g)
            else:
                nvec = torch.randn((), device=scores.device, dtype=scores.dtype)
            scores[i, self.latent_pad_id] = scores[i, self.latent_pad_id] + nvec * self.std
        return scores


def _trl_config_num_generations(global_batch_size: int, user_rollouts: int) -> int:
    """TRL parent check: global_batch % num_generations == 0; take the largest factor that does not exceed user_rollouts."""
    gbs = int(global_batch_size)
    if gbs < 1:
        return 2
    ur = max(2, int(user_rollouts))
    for g in range(min(ur, gbs), 1, -1):
        if gbs % g == 0:
            return g
    for g in range(gbs, 1, -1):
        if gbs % g == 0:
            return g
    return 2


def conversation_to_prompt_only(conversation: Union[List[dict], Any]) -> List[dict]:
    if isinstance(conversation, tuple):
        conversation = conversation[0]
    out = []
    for turn in conversation:
        if turn.get("role") == "assistant":
            break
        out.append(copy.deepcopy(turn))
    if not out:
        raise ValueError("conversation_to_prompt_only: 未找到 user 段")
    return out


def _replace_prompt_pil_images_with_paths(prompt_messages: List[dict], sample: dict) -> List[dict]:
    out = copy.deepcopy(prompt_messages)
    img_in = sample.get("image_input")
    if img_in is None:
        path_iter = iter(())
    elif isinstance(img_in, list):
        path_iter = iter(img_in)
    else:
        path_iter = iter((img_in,))

    def walk_content(items: Any) -> Any:
        if not isinstance(items, list):
            return items
        new_items = []
        for item in items:
            if (
                isinstance(item, dict)
                and item.get("type") == "image"
                and isinstance(item.get("image"), PILImage.Image)
            ):
                p = next(path_iter)
                new_items.append({**item, "image": p})
            else:
                new_items.append(item)
        return new_items

    for turn in out:
        c = turn.get("content")
        if isinstance(c, list):
            turn["content"] = walk_content(c)
    return out


def _sanitize_content_dicts_for_qwen_vision(messages: List[dict]) -> None:
    for turn in messages:
        c = turn.get("content")
        if not isinstance(c, list):
            continue
        for item in c:
            if not isinstance(item, dict):
                continue
            for k in ("image", "image_url", "video"):
                if k in item and item[k] is None:
                    del item[k]


def build_train_dataset(data_path: str, preprocess_function) -> Dataset:
    raw = load_jsonl_dataset(data_path)
    convs = [preprocess_function(s) for s in tqdm(raw, desc="preprocess")]
    rows = []
    for sample, c in tqdm(zip(raw, convs), total=len(raw), desc="build rows"):
        prompt = _replace_prompt_pil_images_with_paths(conversation_to_prompt_only(c), sample)
        rows.append(
            {
                "prompt": prompt,
                "text_output": sample.get("text_output", ""),
                "image_input": sample.get("image_input"),
                "image_output": sample.get("image_output"),
                "map_desc": sample.get("map_desc", ""),
            }
        )
    return Dataset.from_list(
        rows,
        features=Features(
            {
                "prompt": Json(),
                "text_output": Value("string"),
                "image_input": Json(),
                "image_output": Json(),
                "map_desc": Json(),
            }
        ),
    )


def extract_boxed_answer(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    return m.group(1).strip() if m else None


def extract_vsp_path_candidate(text: str) -> str:
    if not text:
        return ""
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed
    m = re.search(r"\{(.*?)\}", text)
    if m:
        return m.group(1)
    return text


def completion_text_from_item(c) -> str:
    if isinstance(c, list) and c and isinstance(c[0], dict):
        return c[0].get("content", "")
    return str(c)


def _traj_pair_avg_cosine(h_i: torch.Tensor, h_j: torch.Tensor) -> float:
    """
    sim(tau_i, tau_j) = average_t cosine(h_t^i, h_t^j), aligned to the shorter latent trajectory.
    h_* : [T, D]; return 0.0 when there is no overlapping length.
    """
    if h_i is None or h_j is None or h_i.numel() == 0 or h_j.numel() == 0:
        return 0.0
    n = int(min(h_i.size(0), h_j.size(0)))
    if n <= 0:
        return 0.0
    a, b = h_i[:n], h_j[:n]
    cos = F.cosine_similarity(a, b, dim=-1, eps=1e-8)
    return float(cos.mean().detach().float().item())


def _latent_traj_contrastive_rewards(
    h_rows: list[torch.Tensor],
    r_ok: list[float],
    temperature: float,
) -> list[float]:
    """
    Rollouts in one group: the first correct sample is tau_target; the incorrect set is tau_neg
    (a negative pair excludes itself when computing sim).
    All-wrong, all-correct, and mixed cases follow the specified formula; tau is the temperature and is clamped away from zero.
    """
    g = len(h_rows)
    tau = max(float(temperature), 1e-8)
    out = [0.0] * g
    if g == 0:
        return out

    def sim(i: int, j: int) -> float:
        return _traj_pair_avg_cosine(h_rows[i], h_rows[j])

    correct = [i for i in range(g) if r_ok[i] >= 0.5]
    wrong = [i for i in range(g) if r_ok[i] < 0.5]

    if not wrong:
        if not correct:
            return [0.0] * g
        t_idx = int(correct[0])
        for i in range(g):
            s = (1 + sim(i, t_idx)) / 2
            out[i] = s / tau
        return out

    if not correct:
        for i in range(g):
            ssum = 0.0
            for j in wrong:
                if j == i:
                    continue
                ssum += math.exp(((1+sim(i, j))/2) / tau)
            # out[i] = math.log(1.0 / (1.0 + ssum))
            out[i] = 1.0 / (1.0 + ssum)
        return out

    t_idx = int(correct[0])
    for i in range(g):
        s_pos = (1 + sim(i, t_idx)) / 2
        pos_term = math.exp(s_pos / tau)
        s_neg = 0.0
        for j in wrong:
            if j == i:
                continue
            s_neg += math.exp(((1+sim(i, j))/2) / tau)
        den = pos_term + s_neg
        if den > 0.0:
            # out[i] = math.log(pos_term / den)
            out[i] = pos_term / den
        else:
            out[i] = -80.0
    return out


def _rows_from_padded_latent(
    h_all: torch.Tensor,
    m_all: torch.Tensor,
    lo: int,
    hi: int,
) -> list[torch.Tensor]:
    """Extract [T_i, D] latent trajectories from the gathered [N, T, D] tensor and mask."""
    h_rows: list[torch.Tensor] = []
    for k in range(lo, hi):
        mrow = m_all[k]
        if bool(mrow.any().item()):
            idx = mrow.nonzero(as_tuple=False).flatten()
            h_rows.append(h_all[k, idx].float())
        else:
            d = h_all.size(-1)
            h_rows.append(h_all[k, :0, :].float() if d else torch.empty(0, 0, device=h_all.device))
    return h_rows


def _latent_traj_sim_rewards_gathered(
    h_comp: torch.Tensor,
    latent_pad_mask: torch.Tensor,
    r_ok_vals: list[float],
    num_generations: int,
    temperature: float,
    device: torch.device,
    accelerator: Any,
) -> Optional[torch.Tensor]:
    """
    Matches TRL GRPO: the local batch is the per-rank subset. The G rollouts of a group are contiguous
    in process order after gather (same order as the gathered rewards). Contrastive rewards are computed
    per group, then only the scalars for this rank's samples are returned, shape [B_local].
    """
    bsz = len(r_ok_vals)
    gsz = int(num_generations)
    if bsz == 0 or gsz < 1:
        return None
    d_h = h_comp.size(-1)

    h_rows_local: list[torch.Tensor] = []
    for bi in range(bsz):
        mrow = latent_pad_mask[bi]
        if bool(mrow.any().item()):
            idx = mrow.nonzero(as_tuple=False).flatten()
            h_rows_local.append(h_comp[bi, idx].detach().float())
        else:
            h_rows_local.append(h_comp[bi, :0, :].detach().float())

    t_max_local = int(max((h.size(0) for h in h_rows_local), default=0))
    t_tensor = torch.tensor([t_max_local], device=device, dtype=torch.long)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(t_tensor, op=torch.distributed.ReduceOp.MAX)
    t_pad = int(t_tensor.item())
    if t_pad <= 0 and d_h > 0:
        t_pad = 1

    if d_h == 0:
        h_pad = torch.empty(bsz, 0, 0, device=device, dtype=torch.float32)
        m_pad = torch.empty(bsz, 0, device=device, dtype=torch.bool)
    else:
        h_pad = torch.zeros(bsz, t_pad, d_h, device=device, dtype=torch.float32)
        m_pad = torch.zeros(bsz, t_pad, device=device, dtype=torch.bool)
        for i, h in enumerate(h_rows_local):
            t = int(min(h.size(0), t_pad)) if h.numel() else 0
            if t > 0:
                h_pad[i, :t] = h[:t]
                m_pad[i, :t] = True

    r_loc = torch.tensor(r_ok_vals, device=device, dtype=torch.float32)
    r_all = gather(r_loc)
    h_st = gather(h_pad)
    m_st = gather(m_pad)
    n_tot = int(r_all.numel())

    if n_tot == 0 or n_tot % gsz != 0:
        logging.warning(
            "use_latent_traj_sim_reward: gather 后总条数=%d 无法整除 G=%d（与 simple 奖励对齐失败），本步已退回。",
            n_tot,
            gsz,
        )
        return None

    vals_out = torch.empty(n_tot, device=device, dtype=torch.float32)
    n_groups = n_tot // gsz
    for g in range(n_groups):
        lo, hi = g * gsz, (g + 1) * gsz
        h_rows = _rows_from_padded_latent(h_st, m_st, lo, hi)
        rok = [float(x) for x in r_all[lo:hi].tolist()]
        rg = _latent_traj_contrastive_rewards(h_rows, rok, temperature)
        for t, rv in enumerate(rg):
            vals_out[lo + t] = float(rv)

    pidx = int(getattr(accelerator, "process_index", 0) or 0)
    return vals_out[pidx * bsz : (pidx + 1) * bsz].contiguous()


def simple_reward_fn(
    prompts,
    completions,
    tokenizer,
    completion_ids=None,
    completion_mask=None,
    text_output=None,
    latent_size: int = 8,
    format_reward_boxed_only: bool = False,
    **kwargs,
):
    """
    Latent format score + \\boxed score + answer match + optional VSP simulation r_ok; compatible with default TRL GRPO.
    When format_reward_boxed_only is True, only \\boxed{} is counted, without the <|latent_pad|> ratio score (Instruct base).
    """
    n = len(completions)
    if text_output is None:
        text_output = [""] * n
    map_desc = kwargs.get("map_desc")
    try:
        latent_pad_id = tokenizer.convert_tokens_to_ids("<|latent_pad|>")
    except Exception:
        latent_pad_id = None

    scores = []
    for i, c in enumerate(completions):
        text = completion_text_from_item(c)
        if completion_ids is not None and latent_pad_id is not None:
            row_ids = completion_ids[i]
            if completion_mask is not None:
                valid = completion_mask[i].bool()
                latent_count = ((row_ids == latent_pad_id) & valid).sum().item()
            else:
                latent_count = (row_ids == latent_pad_id).sum().item()
            r_latent = (
                min(float(latent_count), float(latent_size)) / float(latent_size) * 0.5 if latent_size > 0 else 0.0
            )
        else:
            r_latent = 0.0
        if format_reward_boxed_only:
            r_latent = 0.0
        r_box = 0.5 if re.search(r"\\boxed\{[^}]+\}", text) else 0.0
        pred = extract_boxed_answer(text)
        gt = extract_boxed_answer(text_output[i] if i < len(text_output) else "")
        r_ok = 1.0 if pred is not None and gt is not None and pred == gt else 0.0
        if isinstance(map_desc, list) and i < len(map_desc) and map_desc[i]:
            vsp_path = extract_vsp_path_candidate(text)
            try:
                sim_result = simulate_vsp(map_desc[i], vsp_path)
                r_ok = 1.0 if bool(sim_result.get("success", False)) else 0.0
            except Exception:
                pass
        scores.append(r_latent + r_box + r_ok)
    return scores


# ----- training curves -----


def plot_training_curves(log_history: list[dict[str, Any]], output_png: str, title: Optional[str] = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preferred = [
        "loss",
        "reward",
        "reward_std",
        "kl",
        "policy_entropy",
        "importance_ratio_mean",
        "latent_noise_ratio_mean",
        "rewards/latent_traj_sim_mean",
        "completion_length",
        "learning_rate",
        "grad_norm",
    ]
    series: dict[str, tuple[list[int], list[float]]] = {}
    for entry in log_history:
        st = entry.get("step", entry.get("global_step"))
        if st is None:
            continue
        step = int(st)
        for k, v in entry.items():
            if k in ("step", "epoch", "global_step") or isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                series.setdefault(k, ([], []))[0].append(step)
                series[k][1].append(float(v))

    keys = [k for k, (ss, vv) in series.items() if len(vv) > 0]
    if not keys:
        logging.warning("无可绘制指标，跳过 %s", output_png)
        return
    ordered = [k for k in preferred if k in keys] + sorted(k for k in keys if k not in preferred)

    ncols = 2 if len(ordered) > 6 else 1
    nrows = (len(ordered) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12 if ncols == 2 else 10, max(2.8 * nrows, 4.0)), squeeze=False)
    if title:
        fig.suptitle(title, fontsize=12)
    for idx, key in enumerate(ordered):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        xs, ys = series[key]
        ax.plot(xs, ys, linewidth=1.2)
        if (
            "reward" in key.lower()
            or key.lower().startswith("rewards/")
            or key == "importance_ratio_mean"
            or key == "policy_entropy"
        ) and len(xs) >= 2:
            try:
                import numpy as np

                coef = np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)
                ax.plot(xs, (coef[0] * np.asarray(xs, float) + coef[1]).tolist(), "--", alpha=0.85)
            except Exception:
                pass
        ax.set_xlabel("step")
        display_key = "policy_entropy" if key == "policy_entropy" else key
        ax.set_title(display_key)
        ax.grid(True, alpha=0.3)
    for j in range(len(ordered), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png)) or ".", exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("已保存曲线: %s", output_png)


def plot_r_ok_and_latent_ratio_curves(
    log_history: list[dict[str, Any]],
    output_png: str,
    title: Optional[str] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ("rewards/r_ok_mean", "rewards/r_latent_ratio_mean", "rewards/r_div_mean")
    series: dict[str, tuple[list[int], list[float]]] = {k: ([], []) for k in keys}
    for entry in log_history:
        st = entry.get("step", entry.get("global_step"))
        if st is None:
            continue
        step = int(st)
        for k in keys:
            v = entry.get(k)
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                series[k][0].append(step)
                series[k][1].append(float(v))

    if not any(len(series[k][1]) > 0 for k in keys):
        logging.warning("无 r_ok / r_latent_ratio 指标，跳过 %s", output_png)
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), squeeze=False)
    if title:
        fig.suptitle(title, fontsize=12)
    keys_two = ("rewards/r_ok_mean", "rewards/r_latent_ratio_mean")
    titles = ("rewards/r_ok_mean", "rewards/r_latent_ratio_mean")
    for ax, key, ax_title in zip(axes[0], keys_two, titles):
        xs, ys = series[key]
        if ys:
            ax.plot(xs, ys, linewidth=1.2, color="C0")
            if len(xs) >= 2:
                try:
                    import numpy as np

                    coef = np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)
                    ax.plot(xs, (coef[0] * np.asarray(xs, float) + coef[1]).tolist(), "--", alpha=0.85)
                except Exception:
                    pass
        else:
            ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_xlabel("step")
        ax.set_title(ax_title)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png)) or ".", exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("已保存 r_ok / r_latent_ratio 曲线: %s", output_png)


class PlotCurvesCallback(TrainerCallback):
    def __init__(self, output_png: str, title: Optional[str] = None, every_n_steps: int = 10):
        self.output_png = output_png
        self.title = title
        self.every_n_steps = max(1, int(every_n_steps))
        if output_png:
            root, ext = os.path.splitext(output_png)
            self.detail_output_png = f"{root}_r_ok_r_latent_ratio{ext or '.png'}"
        else:
            self.detail_output_png = None

    def on_step_end(self, args, state, control, **kwargs):
        if int(getattr(args, "process_index", 0) or 0) != 0:
            return control
        step = int(getattr(state, "global_step", 0) or 0)
        if step <= 0 or step % self.every_n_steps != 0:
            return control
        try:
            plot_training_curves(state.log_history, self.output_png, title=self.title)
            if self.detail_output_png:
                plot_r_ok_and_latent_ratio_curves(
                    state.log_history, self.detail_output_png, title=self.title
                )
        except Exception:
            logging.exception("更新曲线失败 step=%s", step)
        return control


class LatentGRPOTrainer(GRPOTrainer):
    """
    Qwen2.5-VL multimodal: _encode_vl_prompt, _get_per_token_logps, _prepare_inputs;
    default matches TRL GRPO; optional latent-noise: see --use_latent_noise_ratio.
    """

    def __init__(
        self,
        model,
        reward_funcs,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        processor=None,
        reward_processing_classes=None,
        callbacks=None,
        optimizers=(None, None),
        peft_config=None,
        use_latent_noise_ratio: bool = False,
        latent_h_noise_std: float = 0.1,
        latent_noise_rollout_logit_std: float = 0.0,
        use_latent_traj_sim_reward: bool = False,
        latent_traj_sim_temp: float = 1.0,
    ):
        if processor is None:
            raise ValueError("需要 processor=AutoProcessor")
        self.mm_processor = processor
        self._use_latent_noise_ratio = bool(use_latent_noise_ratio)
        self._latent_h_noise_std = float(latent_h_noise_std)
        self._latent_noise_rollout_logit_std = float(latent_noise_rollout_logit_std)
        self._use_latent_traj_sim_reward = bool(use_latent_traj_sim_reward)
        self._latent_traj_sim_temp = float(latent_traj_sim_temp)
        if not hasattr(model, "warnings_issued") or model.warnings_issued is None:
            model.warnings_issued = {}
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processor.tokenizer,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        self._current_pixel_values: Optional[torch.Tensor] = None
        self._current_image_grid_thw: Optional[torch.Tensor] = None

    def _encode_vl_prompt(self, prompt_messages_batch: list[list[dict]]):
        prompt_messages_batch = copy.deepcopy(prompt_messages_batch)
        for msgs in prompt_messages_batch:
            _sanitize_content_dicts_for_qwen_vision(msgs)
        texts = []
        for msgs in prompt_messages_batch:
            t = self.mm_processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            t = place_input_image(t)
            texts.append(t)
        image_inputs, _ = process_vision_info(prompt_messages_batch)
        return self.mm_processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        kwargs: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
        pv, ig = self._current_pixel_values, self._current_image_grid_thw
        if pv is not None:
            kwargs["pixel_values"] = pv
        if ig is not None:
            kwargs["image_grid_thw"] = ig
        outputs = model(**kwargs)
        logits = outputs.logits[:, :-1, :]
        tail_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:, :]
        return selective_log_softmax(logits, tail_ids)

    def _prepare_inputs(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        if self.args.use_vllm:
            raise NotImplementedError("多模态请设置 use_vllm=False")

        device = self.accelerator.device
        try:
            lpid = int(self.processing_class.convert_tokens_to_ids("<|latent_pad|>"))
        except Exception:
            lpid = -1
        prompts = [copy.deepcopy(x["prompt"]) for x in inputs]

        vl_batch = self._encode_vl_prompt([copy.deepcopy(p) for p in prompts])
        vl_batch = HfTrainer._prepare_inputs(self, vl_batch)
        prompt_ids = vl_batch["input_ids"]
        prompt_mask = vl_batch["attention_mask"]
        pixel_values = vl_batch.get("pixel_values")
        image_grid_thw = vl_batch.get("image_grid_thw")
        self._current_pixel_values = pixel_values
        self._current_image_grid_thw = image_grid_thw

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        prompts_text = [maybe_apply_chat_template({"prompt": p}, self.processing_class)["prompt"] for p in prompts]

        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            gen_kwargs: dict[str, Any] = {
                "input_ids": prompt_ids,
                "attention_mask": prompt_mask,
                "generation_config": self.generation_config,
            }
            if pixel_values is not None:
                gen_kwargs["pixel_values"] = pixel_values
            if image_grid_thw is not None:
                gen_kwargs["image_grid_thw"] = image_grid_thw
            if (
                self._use_latent_noise_ratio
                and self._latent_noise_rollout_logit_std > 0.0
                and lpid >= 0
            ):
                gen_kwargs["logits_processor"] = LogitsProcessorList(
                    [LatentPadLogitNoiseProcessor(lpid, self._latent_noise_rollout_logit_std, None)]
                )
            prompt_completion_ids = unwrapped_model.generate(**gen_kwargs)

        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        latent_h_noise: Optional[torch.Tensor] = None
        latent_pad_mask: Optional[torch.Tensor] = None
        h_comp: Optional[torch.Tensor] = None
        need_hidden = lpid is not None and lpid >= 0 and int(logits_to_keep) > 0 and (
            self._use_latent_noise_ratio or self._use_latent_traj_sim_reward
        )
        if need_hidden:
            latent_pad_mask = completion_ids == int(lpid)
            um = self.accelerator.unwrap_model(self.model)
            was_gc = getattr(um.model, "gradient_checkpointing", False)
            if was_gc:
                um.model.gradient_checkpointing = False
            try:
                out_ref = um(
                    input_ids=prompt_completion_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    output_hidden_states=True,
                    return_dict=True,
                )
            finally:
                if was_gc:
                    um.model.gradient_checkpointing = True
            # On this repo's Qwen2.5-VL, hidden_states is already the last layer [B, S, D]
            hseq = out_ref.hidden_states
            pln = int(prompt_length)
            h_comp = hseq[:, pln : pln + int(logits_to_keep), :]
            if self._use_latent_noise_ratio and bool(latent_pad_mask is not None and latent_pad_mask.any().item()):
                latent_h_noise = torch.zeros_like(h_comp)
                m = latent_pad_mask
                h_comp_det = h_comp[m].detach()
                h_noise = h_comp_det + self._latent_h_noise_std * torch.randn(
                    h_comp_det.shape, device=h_comp.device, dtype=h_comp.dtype
                )
                latent_h_noise[m] = h_noise

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        text_outputs = [x.get("text_output", "") for x in inputs]
        r_ok_vals: list[float] = []
        for i, c in enumerate(completions):
            text = completion_text_from_item(c)
            pred = extract_boxed_answer(text)
            gt = extract_boxed_answer(text_outputs[i] if i < len(text_outputs) else "")
            r_ok = 1.0 if pred is not None and gt is not None and pred == gt else 0.0
            if i < len(inputs) and inputs[i].get("map_desc"):
                vsp_path = extract_vsp_path_candidate(text)
                try:
                    sim_result = simulate_vsp(inputs[i]["map_desc"], vsp_path)
                    r_ok = 1.0 if bool(sim_result.get("success", False)) else 0.0
                except Exception:
                    pass
            r_ok_vals.append(r_ok)
        n_m = max(len(r_ok_vals), 1)
        self._metrics.setdefault("rewards/r_ok_mean", []).append(sum(r_ok_vals) / n_m)

        traj_sim_rewards: Optional[torch.Tensor] = None
        if self._use_latent_traj_sim_reward and h_comp is not None and latent_pad_mask is not None:
            bsz = len(prompts)
            traj_t = _latent_traj_sim_rewards_gathered(
                h_comp,
                latent_pad_mask,
                r_ok_vals,
                int(self.num_generations),
                self._latent_traj_sim_temp,
                device,
                self.accelerator,
            )
            if traj_t is not None and traj_t.numel() == bsz:
                traj_sim_rewards = traj_t
                self._metrics.setdefault("rewards/latent_traj_sim_mean", []).append(
                    float(traj_t.float().mean().item())
                )
            elif self._use_latent_traj_sim_reward:
                logging.warning(
                    "use_latent_traj_sim_reward 本步未得到有效张量，已退回 simple 奖励。",
                )

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                keys = [key for key in inputs[0] if key not in ("prompt", "completion")]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                reward_kwargs["tokenizer"] = self.processing_class
                reward_kwargs["completion_ids"] = completion_ids
                reward_kwargs["completion_mask"] = completion_mask
                out = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(out, dtype=torch.float32, device=device)
                if (
                    self._use_latent_traj_sim_reward
                    and traj_sim_rewards is not None
                    and i == 0
                ):
                    rewards_per_func[:, i] = traj_sim_rewards

        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        # print("================================================")
        # print(f"rewards: {mean_grouped_rewards}")
        # if self._use_latent_traj_sim_reward:
        #     print(f"traj_sim_rewards: {traj_sim_rewards}")
        # print(f"advantages: {advantages}")
        # print("================================================")

        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = getattr(reward_func, "__name__", type(reward_func).__name__)
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd
            import wandb

            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)
            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})

        out_batch: dict[str, Any] = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        if self._use_latent_noise_ratio and latent_h_noise is not None and latent_pad_mask is not None:
            out_batch["latent_h_noise"] = latent_h_noise
            out_batch["latent_pad_mask"] = latent_pad_mask
        return out_batch

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._current_pixel_values = inputs.get("pixel_values")
        self._current_image_grid_thw = inputs.get("image_grid_thw")
        lhn = inputs.get("latent_h_noise")
        lpm = inputs.get("latent_pad_mask")
        if not self._use_latent_noise_ratio or lhn is None or lpm is None or not bool(lpm.any().item()):
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch=num_items_in_batch)
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = int(completion_ids.size(1))

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        advantages = inputs["advantages"]
        ratio = torch.exp(per_token_logps - per_token_logps.detach())

        # Run another forward pass to obtain h_now
        um = self.accelerator.unwrap_model(model)
        pv, ig = inputs.get("pixel_values"), inputs.get("image_grid_thw")
        was_gc = getattr(um.model, "gradient_checkpointing", False)
        if was_gc:
            um.model.gradient_checkpointing = False
        try:
            out2 = um(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pv,
                image_grid_thw=ig,
                output_hidden_states=True,
                return_dict=True,
            )
        finally:
            if was_gc:
                um.model.gradient_checkpointing = True
        hseq2 = out2.hidden_states  # same as _prepare_inputs: this is the last hidden state
        pln = int(prompt_ids.size(1))
        h_now = hseq2[:, pln : pln + logits_to_keep, :]
        cos = F.cosine_similarity(h_now, lhn.to(h_now.dtype), dim=-1, eps=1e-8)
        r_latent = torch.exp(-0.5 * cos)
        ratio = torch.where(lpm, r_latent, ratio)

        per_token_loss = ratio * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp_min(1)).mean()

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp_min(1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        with torch.no_grad():
            m = lpm
            if m.any():
                r_mean = float(r_latent[m].float().mean().detach().cpu().item())
                self._metrics.setdefault("latent_noise_ratio_mean", []).append(r_mean)
        return loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--load_model_path", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--save_model_path", type=str, required=True)
    p.add_argument("--task", type=str, default="vsp-spatial-reasoning")
    p.add_argument("--latent_size", type=int, default=8)
    p.add_argument("--compress_strategy", type=str, default="average")
    p.add_argument("--progressive_strategy", type=str, default="meanPool")
    p.add_argument("--proj_strategy", type=str, default="None")
    p.add_argument("--contrastive_strategy", type=str, default="None")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--max_prompt_length", type=int, default=2048)
    p.add_argument("--max_completion_length", type=int, default=1024)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache_dir", type=str, default="/data/exp/.cache")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_run_name", type=str, default="grpo_latent_easy")
    p.add_argument("--plot_output_dir", type=str, default=None)
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--plot_frequency", type=int, default=10)
    p.add_argument("--accelerate", action="store_true")
    p.add_argument("--max_gpu_memory", type=float, default=1.0)
    p.add_argument(
        "--use_latent_noise_ratio",
        action="store_true",
        help="在 <|latent_pad|> 用 r=exp(-0.5*cos(h_now, h_noise))；h_noise 在 rollout 序列上做 last-hidden 在 latent 位加高斯后 detach。",
    )
    p.add_argument(
        "--latent_h_noise_std",
        type=float,
        default=0.1,
        help="构造 h_noise 时相对 rollout hidden 的加性高斯标准差。",
    )
    p.add_argument(
        "--latent_noise_rollout_logit_std",
        type=float,
        default=0.0,
        help=">0 时 generate 在每一步对 <|latent_pad|> 词表 logit 加高斯，使多 rollout 采到不同 latent。",
    )
    p.add_argument(
        "--use_latent_traj_sim_reward",
        action="store_true",
        help="以 <|latent_pad|> 位 last hidden 轨迹的 mean_t cos 为 sim(τ_i,τ_j)，按组内对/错做对比 log-softmax 取代 simple_reward 标量。",
    )
    p.add_argument(
        "--latent_traj_sim_temp",
        type=float,
        default=1.0,
        help="latent 轨迹对比奖励中 softmax 的温度 τ（与 --temperature 生成分开）。",
    )
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.environ.setdefault("HF_HOME", args.cache_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    accelerator = None
    if args.accelerate:
        torch.cuda.set_per_process_memory_fraction(args.max_gpu_memory)
        if args.use_wandb:
            from accelerate import Accelerator

            accelerator = Accelerator(log_with="wandb")
            accelerator.init_trackers(
                project_name="latent-grpo-easy",
                config={"task": args.task, "epochs": args.epochs, "batch_size": args.per_device_train_batch_size},
                init_kwargs={"wandb": {"name": args.wandb_run_name}},
            )
    elif args.use_wandb:
        import wandb

        wandb.init(project="latent-grpo-easy", name=args.wandb_run_name, reinit=True)

    world = int(os.environ.get("WORLD_SIZE", "1"))
    global_bs = args.per_device_train_batch_size * world
    trl_num_gen = _trl_config_num_generations(global_bs, args.num_generations)
    if trl_num_gen != int(args.num_generations):
        logging.warning(
            "TRL 要求 global_batch 可被 num_generations 整除：已将 GRPOConfig.num_generations 设为 %s（你请求的 G=%s）",
            trl_num_gen,
            args.num_generations,
        )
    logging.info(
        "TRL-default GRPO | accelerate=%s 目标每 prompt rollout=%s global_batch=%d 实际 num_generations=%s",
        args.accelerate,
        args.num_generations,
        global_bs,
        trl_num_gen,
    )
    if world > 1 and not args.accelerate:
        logging.warning("WORLD_SIZE>1 建议 --accelerate + accelerate launch")

    train_ds = build_train_dataset(args.data_path, task_preporcess_config[args.task])
    logging.info("num_samples=%d", len(train_ds))

    processor = AutoProcessor.from_pretrained(args.load_model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"

    cfg = Qwen2_5_VLConfig.from_pretrained(args.load_model_path)
    cfg.compress_strategy = args.compress_strategy
    cfg.progressive_strategy = args.progressive_strategy
    cfg.proj_strategy = args.proj_strategy
    cfg.contrastive_strategy = args.contrastive_strategy
    cfg.latent_size = args.latent_size
    cfg.stage = "stage3-progressive-grpo"
    cfg.do_sample = True

    load_kw = dict(config=cfg, torch_dtype=torch.bfloat16)
    if args.accelerate:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.load_model_path, **load_kw)
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.load_model_path, **load_kw, device_map="auto"
        )

    # format_reward_boxed_only = (
    #     os.path.basename(os.path.normpath(args.load_model_path.rstrip("/"))) == "Qwen2.5-VL-7B-Instruct"
    # )
    format_reward_boxed_only = "qwen2_5-vl-7b" in args.load_model_path or "Qwen2.5-VL-7B-Instruct" in args.load_model_path

    tok = processor.tokenizer
    if not format_reward_boxed_only:
        model.config.latent_token_id = int(tok("<|latent_pad|>", return_tensors="pt")["input_ids"][0])
        model.config.latent_start_id = int(tok("<|latent_start|>", return_tensors="pt")["input_ids"][0])
        model.config.latent_end_id = int(tok("<|latent_end|>", return_tensors="pt")["input_ids"][0])

    for p in model.visual.parameters():
        p.requires_grad = False

    if format_reward_boxed_only:
        logging.info("Qwen2.5-VL-7B-Instruct 基座：格式奖励仅计 \\boxed{}，不含 <|latent_pad|> 格式分")

    if args.use_latent_noise_ratio:
        logging.info(
            "use_latent_noise_ratio | latent_h_noise_std=%s latent_noise_rollout_logit_std=%s",
            args.latent_h_noise_std,
            args.latent_noise_rollout_logit_std,
        )
    if getattr(args, "use_latent_traj_sim_reward", False):
        logging.info(
            "use_latent_traj_sim_reward | latent_traj_sim_temp=%s",
            getattr(args, "latent_traj_sim_temp", 1.0),
        )

    reward_fn = partial(
        simple_reward_fn,
        tokenizer=processor.tokenizer,
        latent_size=args.latent_size,
        format_reward_boxed_only=format_reward_boxed_only,
    )

    grpo_common = dict(
        output_dir=args.save_model_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_generations=trl_num_gen,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        temperature=args.temperature,
        logging_steps=args.logging_steps,
        bf16=True,
        remove_unused_columns=False,
        report_to=(["wandb"] if args.use_wandb else []),
        max_grad_norm=1.0,
        logging_strategy="steps",
        save_strategy="no",
        logging_dir="./logs/",
        use_vllm=False,
    )
    if args.accelerate:
        grpo_args = GRPOConfig(
            **grpo_common,
            ddp_find_unused_parameters=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        grpo_args = GRPOConfig(**grpo_common)

    if args.plot_output_dir:
        os.makedirs(args.plot_output_dir, exist_ok=True)
    curve = os.path.join(args.plot_output_dir, "training_curves.png") if args.plot_output_dir else None
    callbacks = (
        [PlotCurvesCallback(curve, title=args.wandb_run_name, every_n_steps=max(1, args.plot_frequency))]
        if not args.no_plot
        else None
    )

    trainer = LatentGRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_args,
        train_dataset=train_ds,
        processor=processor,
        callbacks=callbacks,
        use_latent_noise_ratio=getattr(args, "use_latent_noise_ratio", False),
        latent_h_noise_std=getattr(args, "latent_h_noise_std", 0.1),
        latent_noise_rollout_logit_std=getattr(args, "latent_noise_rollout_logit_std", 0.0),
        use_latent_traj_sim_reward=getattr(args, "use_latent_traj_sim_reward", False),
        latent_traj_sim_temp=getattr(args, "latent_traj_sim_temp", 1.0),
    )

    _gc = trainer.generation_config
    _gc.temperature = float(args.temperature)
    _gc.do_sample = float(args.temperature) > 0.0
    if getattr(trainer.accelerator, "is_main_process", True):
        logging.info("rollout generation_config: temperature=%s do_sample=%s", _gc.temperature, _gc.do_sample)

    trainer.train()

    if getattr(trainer, "is_fsdp_enabled", False):
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model(args.save_model_path)
    processor.save_pretrained(args.save_model_path)

    if not args.no_plot and getattr(trainer.accelerator, "is_main_process", True):
        try:
            plot_training_curves(trainer.state.log_history, curve, title=args.wandb_run_name)
            if curve:
                root, ext = os.path.splitext(curve)
                plot_r_ok_and_latent_ratio_curves(
                    trainer.state.log_history,
                    f"{root}_r_ok_r_latent_ratio{ext or '.png'}",
                    title=args.wandb_run_name,
                )
        except Exception:
            logging.exception("最终绘图失败")

    if args.accelerate and accelerator is not None:
        accelerator.end_training()
    elif args.use_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
