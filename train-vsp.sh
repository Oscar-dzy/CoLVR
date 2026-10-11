# export CUDA_VISIBLE_DEVICES=6
export CUDA_LAUNCH_BLOCKING=1
export WANDB_API_KEY=your_wandb_api_key
export WANDB_MODE=offline

ACCELERATE_CONFIG_FILE=./config/default_config.yaml


export CUDA_VISIBLE_DEVICES=0,1,2,3
# Training stage 1-progressive
accelerate launch --config_file $ACCELERATE_CONFIG_FILE \
    src/main.py \
    --model                  /path/to/Qwen2.5-VL-7B-Instruct \
    --epochs                 10 \
    --lr                     5e-5 \
    --task                   vsp-spatial-planning-progressive \
    --latent_size            8 \
    --stage                  stage1-progressive \
    --progressive_strategy   meanPool \
    --data_path              ./data/vsp_spatial_planning/train_direct.jsonl \
    --log_file               ./log.txt \
    --save_model_path        /path/to/save/CoLVR-warmup-VSP-checkpoints \
    --wandb_run_name         vsp-stage1-prog-meanPool-lr5e-5-10epochs \
    --accelerate \



# Training stage 2-progressive-mean
export CUDA_VISIBLE_DEVICES=0,1,2,3

# python src/main.py \
accelerate launch --config_file $ACCELERATE_CONFIG_FILE \
    src/main.py  \
    --model                  /path/to/Qwen2.5-VL-7B-Instruct \
    --epochs                 10 \
    --lr                     5e-5 \
    --task                   vsp-spatial-planning-progressive \
    --latent_size            8 \
    --stage                  stage2-progressive \
    --progressive_strategy   meanPool \
    --data_path              ./data/vsp_spatial_planning/train_direct.jsonl \
    --log_file               ./log_1.txt \
    --load_model_path        /path/to/save/CoLVR-warmup-VSP-checkpoints \
    --save_model_path        /path/to/save/CoLVR-SFT-VSP-checkpoints  \
    --wandb_run_name         vsp-stage2-prog-meanPool-lr5e-5-10epochs \
    --accelerate \