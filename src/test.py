import torch
from typing import Optional
from collections import defaultdict
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLConfig
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


seed_everything(seed=42)
args=get_args()



def extract_boxed_format(ans: str) -> Optional[str]:
    """
    Extract the string inside ``\\boxed{...}``. By default, use the **first**
    match (consistent with the previous behavior).

    When ``--eval_cot`` is enabled, use the **last** match, so earlier reasoning
    may contain multiple steps while the final answer appears in the trailing
    ``\\boxed{}``.
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
        if args.task == "vsp-spatial-planning":
            conversations[-1]['content'][1]['text'] = conversations[-1]['content'][1]['text'] + " " + "Let's think step by step."
        else:
            raise ValueError(f"Task {args.task} eval_cot not supported")
            
        # print(conversations[-1]['content'][1]['text'])
        # print("="*100)

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

    # generated_ids_trimmed = [
    #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    # ]
    # answer = processor.batch_decode(
    #     generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    # )[0]

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



def main():
    if args.task == "vsp-spatial-planning":
        eval_vsp()
    else:
        raise ValueError(f"Task {args.task} not supported")


if __name__ == "__main__":
    main()





