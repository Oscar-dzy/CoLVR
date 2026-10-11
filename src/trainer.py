from trl import SFTTrainer, SFTConfig
import torch
import random
from packaging import version
from accelerate import __version__ as accelerate_version
from transformers import AutoProcessor
from transformers.utils import is_torch_xla_available, is_sagemaker_mp_enabled
from PIL import Image

import torch.nn.functional as F

from utils import get_args
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union


args=get_args()
latent_size = args.latent_size

processor = AutoProcessor.from_pretrained(args.model)
processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)

latent_token_id = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]

ALL_IMAGE_OUTPUT_PATHS: List[str] = []



class CustomTrainerStage1(SFTTrainer):
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        (ce_loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        predict_embeddings = outputs.hidden_states
        image_out_mask = inputs["image_out_mask"]  # stage 1 implicitly aligns the model to this latent token

        shift_image_mask = image_out_mask[:, -(predict_embeddings.shape[1] - 1) :].to(predict_embeddings.device)
        shift_predict_embeddings = predict_embeddings[..., :-1, :][shift_image_mask.to(predict_embeddings.device) != 0].contiguous()

        input_embeddings = outputs.inputs_embeds
        gt_embeddings = input_embeddings[..., 1:, :][shift_image_mask.to(input_embeddings.device) != 0].contiguous()
        
        sim_loss = F.cosine_similarity(gt_embeddings, shift_predict_embeddings).mean()
        sim_loss = 1 - sim_loss

        loss = 0.1 * ce_loss + sim_loss
        return (loss, outputs) if return_outputs else loss

        

class CustomTrainerStage1_progressive(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        (ce_loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        # start-state mean pool
        startState_embeds_init = outputs.startState_embeds_init
        # latent embeds
        # Hidden states are shifted by one relative to input_ids. The hidden state at token A predicts the next token, B.
        predict_embeddings = outputs.hidden_states
        image_out_mask = inputs["image_out_mask"]
        shift_image_mask = image_out_mask[:, -(predict_embeddings.shape[1] - 1) :].to(predict_embeddings.device)  # shift alignment: hidden_states[:, t] corresponds to image_out_mask[:, t+1]
        shift_predict_embeddings = predict_embeddings[..., :-1, :][shift_image_mask.to(predict_embeddings.device) != 0].contiguous()

        sim_cos_latents_mean = F.cosine_similarity(startState_embeds_init, shift_predict_embeddings).mean()

        sim_loss = 1 - sim_cos_latents_mean

        loss = 0.3 * ce_loss + sim_loss

        if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
            log_mess = {
                "loss_total": loss.detach().cpu().item(),
                "sim_loss": sim_loss.detach().cpu().item(),
                "sim_cos_avg_w_start": sim_cos_latents_mean.detach().cpu().item()  # mean similarity to the start state
            }

            self.log(log_mess)


        return (loss, outputs) if return_outputs else loss

class CustomTrainerStage2(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        (ce_loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        loss = ce_loss
        return (loss, outputs) if return_outputs else loss


class CustomTrainerStage2_progressive_meanPool(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        (ce_loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        
        # start-state / end-state / reference-state mean pool
        startState_embeds = outputs.startState_embeds  # [1, hidden_dim]
        endState_embeds = outputs.endState_embeds
        hidden_dim = startState_embeds.shape[-1]
        startState_embeds = startState_embeds.view(-1, hidden_dim)
        endState_embeds = endState_embeds.view(-1, hidden_dim)
        refState_embeds = (startState_embeds + endState_embeds) / 2
            
        # latent embeds
        predict_embeddings = outputs.hidden_states
        image_out_mask = inputs["image_out_mask"]
        shift_image_mask = image_out_mask[:, -(predict_embeddings.shape[1] - 1) :].to(predict_embeddings.device)  # shift alignment: hidden_states[:, t] predicts labels[:, t+1]
        shift_predict_embeddings = predict_embeddings[..., :-1, :][shift_image_mask.to(predict_embeddings.device) != 0].contiguous()  # [latent_size, hidden_dim]
        # print(f"shift_predict_embeddings.shape[0]:  {shift_predict_embeddings.shape[0]}")
        
        # metrics for visualization
        cos_sim_startStates = []
        cos_sim_endStates = []
        cos_sim_contrastStates = []
        loss_latent = torch.tensor(0.0, device=startState_embeds.device, dtype=startState_embeds.dtype)


        for latent_i in range(shift_predict_embeddings.shape[0]):
            hidden_states = shift_predict_embeddings[latent_i, :]
            hidden_states = hidden_states.view(-1, hidden_dim)

            cos_sim_start = F.cosine_similarity(startState_embeds.detach(), hidden_states).mean()  # start_state or hidden_states: [1, hidden_dim]
            cos_sim_end = F.cosine_similarity(endState_embeds.detach(), hidden_states).mean()  # larger cosine similarity means more similar
            cos_sim_ref = F.cosine_similarity(refState_embeds.detach(), hidden_states).mean()

            # Build N negative cases for each latent token and use InfoNCE loss
            # L = -1/N_latent * sum_i log(exp(sim(latent_i, end)) / (exp(sim(latent_i, end)) + sum_j exp(sim(latent_i, neg_j))))
            cos_sim_startStates.append(cos_sim_start)
            cos_sim_endStates.append(cos_sim_end)

            N_neg = getattr(args, 'N_neg', 8)
            # print(f"N_neg: {N_neg}")
            tau = getattr(args, 'tau', 0.07)
            min_angle = getattr(args, 'min_angle', 90)
            max_angle = getattr(args, 'max_angle', 180)
            min_angle = min_angle * torch.pi / 180
            max_angle = max_angle * torch.pi / 180
            delta_start_end = endState_embeds - startState_embeds
            delta_start_end = delta_start_end.view(-1, hidden_dim)
            delta_start_end = delta_start_end.detach()
            delta_norm = delta_start_end / (delta_start_end.norm(dim=-1, keepdim=True).clamp(min=1e-8))
            delta_norm_sq = (delta_start_end * delta_start_end).sum(dim=-1, keepdim=True).clamp(min=1e-8)

            sim_pos = F.cosine_similarity(endState_embeds.detach(), hidden_states).mean()
            sim_neg_list = []
            for _ in range(N_neg):
                noise = torch.randn_like(delta_start_end)
                epsilon = noise - (noise * delta_start_end).sum(dim=-1, keepdim=True) / delta_norm_sq * delta_start_end
                epsilon_norm = epsilon / (epsilon.norm(dim=-1, keepdim=True).clamp(min=1e-8))
                theta = torch.rand(delta_start_end.size(0), 1, device=delta_start_end.device) * (max_angle - min_angle) + min_angle
                cos_theta = torch.cos(theta)
                sin_theta = torch.sin(theta)
                epsilon = (cos_theta * delta_norm + sin_theta * epsilon_norm) * (delta_start_end.norm(dim=-1, keepdim=True).clamp(min=1e-8))
                neg_embeds = startState_embeds + delta_start_end + epsilon
                neg_embeds = neg_embeds.view(-1, hidden_dim)
                sim_neg = F.cosine_similarity(neg_embeds.detach(), hidden_states).mean()
                sim_neg_list.append(sim_neg)

            cos_sim_contrast = torch.stack(sim_neg_list).mean()  # for visualization: mean similarity to negatives
            cos_sim_contrastStates.append(cos_sim_contrast)

            # InfoNCE: L_i = -log(exp(sim_pos)/tau / (exp(sim_pos)/tau + sum_j exp(sim_neg_j)/tau))
            # use logsumexp for numerical stability
            all_sims = torch.cat([sim_pos.unsqueeze(0), torch.stack(sim_neg_list)])
            loss_i = -sim_pos/tau + torch.logsumexp(all_sims/tau, dim=0)
            loss_latent += loss_i

        # InfoNCE term L = -1/N_latent * sum_i log(...)
        n_latent = shift_predict_embeddings.shape[0]
        loss_latent = loss_latent / n_latent

        loss = ce_loss + 2 * loss_latent

        if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
            log_mess = {}

            for i in range(len(cos_sim_startStates)):
                log_mess[f"cos_sim_start-{i}-Latent"] = cos_sim_startStates[i].detach().cpu().item()
                log_mess[f"cos_sim_end-{i}-Latent"] = cos_sim_endStates[i].detach().cpu().item()
                
                log_mess[f"cos_sim_contrast-{i}-Latent"] = cos_sim_contrastStates[i].detach().cpu().item()

            # TRL SFTTrainer.log raises StopIteration on an empty dict via next(iter(logs.keys()))
            if log_mess:
                self.log(log_mess)


        return (loss, outputs) if return_outputs else loss