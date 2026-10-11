export CUDA_LAUNCH_BLOCKING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export WANDB_API_KEY=wandb_your_api_key
export WANDB_MODE=offline

ACCELERATE_CONFIG_FILE=./config/default_config.yaml



# Training stage 1-progressive
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Training stage 1-progressive
accelerate launch --config_file $ACCELERATE_CONFIG_FILE \
    src/main.py \
    --model                  /path/to/Qwen2.5-VL-7B-Instruct \
    --epochs                 8 \
    --lr                     1e-5 \
    --task                   tertis-progressive \
    --latent_size            8 \
    --stage                  stage1-progressive \
    --progressive_strategy   meanPool \
    --data_path              ./data/zebra_tertis/train_direct.jsonl \
    --log_file               ./log.txt \
    --save_model_path        /path/to/save/CoLVR-warmup-Tertis-checkpoints \
    --wandb_run_name         tertis-stage1-prog-meanPool-warmup \
    --accelerate \


# Training stage 2-progressive-mean
export CUDA_VISIBLE_DEVICES=0,1,2,3

# python src/main.py \
accelerate launch --config_file $ACCELERATE_CONFIG_FILE \
    src/main.py  \
    --model                  /path/to/Qwen2.5-VL-7B-Instruct \
    --epochs                 8 \
    --lr                     1e-5 \
    --task                   tertis-progressive \
    --latent_size            8 \
    --stage                  stage2-progressive \
    --progressive_strategy   meanPool \
    --data_path              ./data/zebra_tertis/train_direct.jsonl \
    --log_file               ./log_1.txt \
    --load_model_path        /path/to/save/CoLVR-warmup-Tertis-checkpoints \
    --save_model_path        /path/to/save/CoLVR-SFT-Tertis-checkpoints \
    --wandb_run_name         tertis-stage2-prog-meanPool-lr1e-5-8epochs \
    --accelerate \