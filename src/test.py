import torch
from typing import Optional
from collections import defaultdict
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLConfig, AutoModel, AutoTokenizer
import os
import re
import json
import logging
from tqdm import tqdm
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

from qwen_vl_utils import process_vision_info
from mathruler.grader import extract_boxed_content

from utils import *
from task import *


seed_everything(seed=42)  # 50
args=get_args()


def extract_boxed_format(ans: str) -> Optional[str]:
    """
    Extract the string inside ``\\boxed{...}``. By default take the **first** match (same as the previous behavior).
    When ``--eval_cot`` is set, take the **last** match so earlier spans can hold intermediate reasoning and the final answer sits in the trailing ``\\boxed{}``.
    """
    if not ans:
        return None
    pat = re.compile(r"\\boxed\{(.*?)\}", re.DOTALL)
    matches = list(pat.finditer(ans))
    if not matches:
        return None
    idx = -1 if getattr(args, "eval_cot", False) else 0
    return matches[idx].group(1)



logging.basicConfig(
    level=logging.INFO,  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S',  # Date format
    handlers=[
        logging.FileHandler(args.log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ],
)

logging.info('=='*20)
logging.info(args)
logging.info('=='*20)
    
# Load the model and processor
cache_dir = args.cache_dir
os.environ['HF_HOME'] = cache_dir

processor = None
config = None
internvl_tokenizer = None

processor = AutoProcessor.from_pretrained(args.load_model_path, trust_remote_code=True, cache_dir=cache_dir)
config = Qwen2_5_VLConfig.from_pretrained(args.load_model_path)
config.proj_strategy = args.proj_strategy
config.stage = "None"


_hf_device_map = "auto"

_load_kw = dict(
    pretrained_model_name_or_path=args.load_model_path,
    device_map=_hf_device_map,
    torch_dtype=torch.bfloat16,
    config=config,
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(**_load_kw)

logging.info("device_map=%s cuda.device_count=%s", _hf_device_map, torch.cuda.device_count())

# processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
# processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
# processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)

latent_pad_id = processor.tokenizer.convert_tokens_to_ids("<|latent_pad|>") if processor is not None else -1

model.eval()

with open(args.data_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]


def run_one_inference(sample):
    preprocess_function = task_test_preporcess_config[args.task]
    conversations = preprocess_function(sample)  # List

    if args.eval_cot:
        if args.task == "tertis":
            conversations[-1]['content'][1]['text'] = conversations[-1]['content'][1]['text'].split("Give only the final option")[0] + "The final option is wrapped in \\boxed{}. Let's think step by step."
        elif args.task == "jigsaw":
            conversations[-1]['content'][1]['text'] = conversations[-1]['content'][1]['text'].split("Give only the final answer")[0] + "The final answer is wrapped in \\boxed{}. Let's think step by step."
        else:
            conversations[-1]['content'][1]['text'] = conversations[-1]['content'][1]['text'] + " " + "Let's think step by step."
        print(conversations[-1]['content'][1]['text'])
        print("="*100)

    texts = [processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)]
    # texts = [place_input_image(text, sep_token=None) for text in texts]
    image_inputs, _ = process_vision_info(conversations)

    inputs = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    grid_thw = inputs['image_grid_thw'][0]
    inputs = inputs.to(model.device)

    hidden_states = None
    attentions = None
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            tokenizer=processor.tokenizer
        )

            
    decoded_output = processor.tokenizer.decode(output_ids[0], skip_special_tokens=False)
    answer = decoded_output.split('<|im_start|>assistant')[-1]

    if args.output_response:
        print(answer)

    return answer, output_ids, hidden_states, attentions, grid_thw


def eval_vsp():
    correct, invalid = 0, 0
    results_level = {}

    for i, sample in tqdm(enumerate(data)):
        if "level7" in sample['image_input'] or "level8" in sample['image_input']:
            level = sample['image_input'].split("/")[-2]
        else:
            level = sample['image_input'].split("/")[-3]
        if level not in results_level:
            results_level[level] = {}
            results_level[level]['correct'] = 0
            results_level[level]['total'] = 0
        
        answer, output_ids, hidden_states, attentions, grid_thw = run_one_inference(sample)

        map_desc = sample.get("map_desc") or []

        path_str = None
        if "boxed" in answer:
            path_str = extract_boxed_format(answer) if args.eval_cot else extract_boxed_content(answer)
        if path_str is None:
            answer_match = re.search(re.compile(r'\{(.*?)\}'), answer)
            if answer_match:
                path_str = answer_match.group(1)
            else:
                path_str = answer
        
        # print(path_str)

        result = simulate_vsp(map_desc, path_str)

        if result['success']: 
            correct += 1
            results_level[level]['correct'] += 1
        elif result['invalid']: 
            invalid += 1

        results_level[level]['total'] += 1

        if (i+1) % 20 == 0:
            logging.info(f"[{i+1}] Accuracy: {correct}/{i+1} ({correct/(i+1):.3f}), Invalid: {invalid}/{i+1} ({invalid/(i+1):.3f})")


    logging.info(f"[Final] Accuracy: {correct}/{i+1} ({correct/(i+1):.3f}), Invalid: {invalid}/{i+1} ({invalid/(i+1):.3f})")

    for level in results_level:
        result = results_level[level]
        logging.info(f"[{level}] Accuracy: {result['correct']}/{result['total']} ({result['correct']/result['total']:.3f})")


def eval_jigsaw():
    total = len(data)
    success_total = 0
    fail_7b_response = 0
    results_cate = {}
    results = []
    
    for i, sample in enumerate(tqdm(data)):
        results_cate = {}
        
        answer, output_ids, hidden_states, attentions, grid_thw = run_one_inference(sample)
        _, grid_h, grid_w = grid_thw

        if "\\boxed" in answer:
            answer_norm = extract_boxed_format(answer)
        else:
            fail_7b_response += 1
            answer_norm = answer

        gt = sample.get("text_output", "")
        if "\\boxed" in gt:
            gt_norm = extract_boxed_format(gt)
        else:
            gt_norm = gt

        # print(f"answer_norm: {answer_norm}")
        # print(f"gt_norm: {gt_norm}")
        ok = (answer_norm == gt_norm)
        if ok: 
            success_total += 1
        
        results.append({
            "index": i,
            "task": args.task,
            "pred": answer,
            "pred_norm": answer_norm,
            "gt": gt,
            "match": bool(ok)
        })

        if (i + 1) % 10 == 0 or (i + 1) == total:
            log_message = f"[{i+1}/{total}] Accuracy: {success_total}/{i+1} ({success_total/(i+1):.4f})"
            if "Qwen2.5-VL-7B" in args.load_model_path:
                 log_message += f" (Fail nums: {fail_7b_response}/{total})"

            logging.info(log_message)
            
    acc = success_total / total
    metrics = {}

    logging.info(f"[Final] Accuracy: {success_total}/{total} ({acc:.4f})")

    metrics["ACC"] = acc

    os.makedirs(args.output_results_dir, exist_ok=True)

    with open(f"{args.output_results_dir}/results.json", "w") as fout:
        json.dump(results, fout, indent=4)

    with open(f"{args.output_results_dir}/metrics.json", "w") as fout:
        json.dump(metrics, fout, indent=4)


def eval_tertis():
    total = len(data)
    success_total = 0
    fail_7b_response = 0
    results_cate = {}
    results = []
    
    for i, sample in enumerate(tqdm(data)):
        results_cate = {}
        
        answer, output_ids, hidden_states, attentions, grid_thw = run_one_inference(sample)
        _, grid_h, grid_w = grid_thw

        if "\\boxed" in answer:
            answer_norm = extract_boxed_format(answer)
        else:
            fail_7b_response += 1
            answer_norm = answer

        gt = sample.get("text_output", "")
        if "\\boxed" in gt:
            gt_norm = extract_boxed_format(gt)
        else:
            gt_norm = gt

        # print(f"answer_norm: {answer_norm}")
        # print(f"gt_norm: {gt_norm}")
        ok = (answer_norm == gt_norm)
        if ok: 
            success_total += 1
        
        results.append({
            "index": i,
            "task": args.task,
            "pred": answer,
            "pred_norm": answer_norm,
            "gt": gt,
            "match": bool(ok)
        })

        if (i + 1) % 10 == 0 or (i + 1) == total:
            log_message = f"[{i+1}/{total}] Accuracy: {success_total}/{i+1} ({success_total/(i+1):.4f})"
            if "Qwen2.5-VL-7B" in args.load_model_path:
                 log_message += f" (Fail nums: {fail_7b_response}/{total})"

            logging.info(log_message)
            
    acc = success_total / total
    metrics = {}

    logging.info(f"[Final] Accuracy: {success_total}/{total} ({acc:.4f})")

    metrics["ACC"] = acc

    
    os.makedirs(args.output_results_dir, exist_ok=True)

    with open(f"{args.output_results_dir}/results.json", "w") as fout:
        json.dump(results, fout, indent=4)

    with open(f"{args.output_results_dir}/metrics.json", "w") as fout:
        json.dump(metrics, fout, indent=4)


def main():
    if args.task == "vsp-spatial-planning":
        eval_vsp()
    elif args.task == "jigsaw":
        eval_jigsaw()
    elif args.task == "tertis":
        eval_tertis()


if __name__ == "__main__":
    main()





