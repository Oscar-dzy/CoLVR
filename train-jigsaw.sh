# export CUDA_VISIBLE_DEVICES=6
export CUDA_LAUNCH_BLOCKING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export WANDB_API_KEY=your_wandb_api_key
export WANDB_MODE=offline

ACCELERATE_CONFIG_FILE=./config/default_config.yaml


# Training stage 1-progressive
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Training stage 1-progressive
accelerate launch --config_file $ACCELERATE_CONFIG_FILE \
    src/main.py \
    --model                  /path/to/Qwen2.5-VL-7B-Instruct \
    --epochs                 8 \
    --lr                     2e-5 \
    --task                   jigsaw-progressive \
    --latent_size            8 \
    --stage                  stage1-progressive \
    --progressive_strategy   meanPool \
    --data_path              ./data/zebra_jigsaw/train_direct.jsonl \
    --log_file               ./log.txt \
    --save_model_path        /path/to/save/CoLVR-warmup-Jigsaw-checkpoints \
    --wandb_run_name         jigsaw-stage1-prog-meanPool-lr2e-5-8epochs-4gpus-cold_start \
    --accelerate \


# Training stage 2-progressive-mean
export CUDA_VISIBLE_DEVICES=0,1,2,3 

# # python src/main.py \
accelerate launch --config_file $ACCELERATE_CONFIG_FILE \
    src/main.py  \
    --model                  /path/to/Qwen2.5-VL-7B-Instruct \
    --epochs                 8 \
    --lr                     2e-5 \
    --task                   jigsaw-progressive \
    --latent_size            8 \
    --stage                  stage2-progressive \
    --progressive_strategy   meanPool \
    --data_path              ./data/zebra_jigsaw/train_direct.jsonl \
    --log_file               ./log_1.txt \
    --load_model_path        /path/to/save/CoLVR-warmup-Jigsaw-checkpoints \
    --save_model_path        /path/to/save/CoLVR-SFT-Jigsaw-checkpoints \
    --wandb_run_name         jigsaw-stage2-prog-meanPool-lr2e-5-8epochs-4gpus \
    --accelerate \