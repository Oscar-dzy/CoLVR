# ============================== VSP Inference ==============================
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_LATENT_FIXED_INFERENCE=true  # Whether to switch to a fixed number of latent reasoning steps; must be set to False for Mirage.
export TRANSFORMERS_MAX_LATENT_LEN=8

data_path=./data/vsp_spatial_planning/test_direct-seen.jsonl  # seen
# data_path=./data/vsp_spatial_planning/test_direct-unseen.jsonl  # unseen

model_path=/path/to/your/CoLVR-checkpoints
# model_path=/path/to/your/Mirage-checkpoints


python src/test.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct --epochs 15 \
    --task vsp-spatial-planning \
    --data_path $data_path \
    --load_model_path  $model_path \
    --output_response \



# ============================== Jigsaw Inference ==============================
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_LATENT_FIXED_INFERENCE=false  # Whether to switch to a fixed number of latent reasoning steps; must be set to False for Mirage.
export TRANSFORMERS_MAX_LATENT_LEN=8

data_path=./data/zebra_jigsaw/test_direct.jsonl

model_path=/path/to/your/CoLVR-checkpoints
# model_path=/path/to/your/Mirage-checkpoints

output_results_dir=/path/to/save_results

cache_dir=/path/to/your/cache

# python src/test.py \
#     --model Qwen/Qwen2.5-VL-7B-Instruct --epochs 15 \
#     --task jigsaw \
#     --data_path $data_path \
#     --load_model_path  $model_path \
#     --cache_dir   $cache_dir  \
#     --output_results_dir $output_results_dir \
#     --output_response \



# ============================== Tertis Inference ==============================
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_LATENT_FIXED_INFERENCE=false  # Whether to switch to a fixed number of latent reasoning steps; must be set to False for Mirage.
export TRANSFORMERS_MAX_LATENT_LEN=8

data_path=./data/zebra_tertis/test_direct.jsonl

model_path=/path/to/your/CoLVR-checkpoints
# model_path=/path/to/your/Mirage-checkpoints

output_results_dir=/path/to/save_results

cache_dir=/path/to/your/cache

# python src/test.py \
#     --model Qwen/Qwen2.5-VL-7B-Instruct --epochs 15 \
#     --task tertis \
#     --data_path $data_path \
#     --load_model_path  $model_path \
#     --cache_dir $cache_dir \
#     --output_results_dir $output_results_dir \
#     --output_response \