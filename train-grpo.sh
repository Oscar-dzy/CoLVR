export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_API_KEY=your_wandb_api_key
# export WANDB_MODE=offline


ACCELERATE_CONFIG_FILE=./config/default_config.yaml

# src/grpo_latent.py \
# VSP
accelerate launch --config_file $ACCELERATE_CONFIG_FILE \
  src/grpo_latent_easy.py \
  --data_path                    ./data/vsp_spatial_planning/grpo_direct_3456.jsonl \
  --task                         vsp-spatial-planning-progressive \
  --load_model_path              /path/to/your/CoLVR-SFT-VSP-checkpoints \
  --save_model_path              /path/to/your/save/CoLVR-GRPO-VSP-checkpoints \
  --plot_output_dir              /path/to/save_metrics_in_GRPO_training \
  --per_device_train_batch_size  1 \
  --num_generations              4 \
  --epochs                       5 \
  --lr                           5e-6 \
  --temperature                  1.2 \
  --max_prompt_length            2048 \
  --max_completion_length        1024 \
  --beta                         0.04 \
  --use_latent_traj_sim_reward   \
  --latent_traj_sim_temp         0.5 \
  --accelerate \
  --max_gpu_memory 1.0

  # --use_latent_traj_sim_reward   \
  # --latent_traj_sim_temp         0.5 \


# Jigsaw
accelerate launch --config_file $ACCELERATE_CONFIG_FILE \
  src/grpo_latent_easy.py \
  --data_path                    ./data/zebra_jigsaw/train_grpo_direct.jsonl \
  --task                         jigsaw-progressive \
  --load_model_path              /path/to/your/CoLVR-SFT-Jigsaw-checkpoints \
  --save_model_path              /path/to/your/CoLVR-GRPO-Jigsaw-checkpoints \
  --plot_output_dir              /path/to/save_metrics_in_GRPO_training \
  --per_device_train_batch_size  1 \
  --num_generations              4 \
  --epochs                       5 \
  --lr                           5e-6 \
  --temperature                  1.2 \
  --max_prompt_length            2048 \
  --max_completion_length        1024 \
  --beta                         0.04 \
  --use_latent_traj_sim_reward   \
  --latent_traj_sim_temp         0.5 \
  --accelerate \
  --max_gpu_memory 1.0



# Tertis
accelerate launch --config_file $ACCELERATE_CONFIG_FILE \
  src/grpo_latent_easy.py \
  --data_path                    ./data/zebra_tertis/train_grpo_direct.jsonl \
  --task                         tertis-progressive \
  --load_model_path              /path/to/your/CoLVR-SFT-Tertis-checkpoints \
  --save_model_path              /path/to/your/CoLVR-GRPO-Tertis-checkpoints \
  --plot_output_dir              /path/to/save_metrics_in_GRPO_training \
  --per_device_train_batch_size  1 \
  --num_generations              4 \
  --epochs                       5 \
  --lr                           5e-6 \
  --temperature                  1.2 \
  --max_prompt_length            2048 \
  --max_completion_length        1024 \
  --beta                         0.04 \
  --use_latent_traj_sim_reward   \
  --latent_traj_sim_temp         0.5 \
  --accelerate \
  --max_gpu_memory 1.0

