# ============================== VSP Inference ==============================
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_LATENT_FIXED_INFERENCE=true  # Whether to switch to a fixed number of latent reasoning steps; must be set to False for Mirage.
export TRANSFORMERS_MAX_LATENT_LEN=8

data_path=./data/CoLVR-VSP_bench/test_direct-seen.jsonl

model_path=/path/to/CoLVR-VSP


python src/test.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct --epochs 15 \
    --task vsp-spatial-planning \
    --data_path $data_path \
    --load_model_path  $model_path \
    --output_response \