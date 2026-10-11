import os
import torch
import wandb
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig, AutoTokenizer, AutoProcessor
from PIL import Image
import os
import logging

from trl import SFTTrainer, SFTConfig
from qwen_vl_utils import process_vision_info

from utils import *
from task import *
from trainer import CustomTrainerStage1, CustomTrainerStage2, CustomTrainerStage1_progressive, CustomTrainerStage2_progressive_meanPool
import trainer as trainer_module

seed_everything(seed=42)
args=get_args()

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:true"
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:2048"

logging.basicConfig(
    level=logging.INFO,  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S',  # Date format
    handlers=[
        logging.FileHandler(args.log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ],
)

if args.accelerate:
    torch.cuda.set_per_process_memory_fraction(args.max_gpu_memory)

    from accelerate import Accelerator
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(
        project_name="latent-qwen2_5-vl-sft", 
        config={
            "model": args.model,
            "epochs": args.epochs,
            "batch_size": 1
        },
        init_kwargs={"wandb": {"name": f"{args.wandb_run_name}"}}
    )

else:
    wandb.init(
        project="latent-qwen2_5-vl-sft", 
        name=f"{args.wandb_run_name}",
        config={
            "model": args.model,
            "epochs": args.epochs,
            "batch_size": 1
        },
        reinit=True
    )

logging.info('=='*20)
logging.info(args)
logging.info('=='*20)

# Load the model and processor
cache_dir = args.cache_dir
os.environ['HF_HOME'] = cache_dir

processor = AutoProcessor.from_pretrained(args.model)

if args.stage not in ['vanilla-sft']:
    processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)


if args.stage in ['stage1-mirage', 'stage1-progressive'] or args.stage in ['vanilla-sft', 'stage1-progressive']: 
    model_path = args.model
    config = Qwen2_5_VLConfig.from_pretrained(model_path)
    grad_checkpointing = True
elif args.stage in ['stage2-mirage', 'stage2-progressive', "stage2-mirage-raw"]:
    model_path = args.load_model_path
    config = Qwen2_5_VLConfig.from_pretrained(model_path)
    grad_checkpointing = False

config.compress_strategy = args.compress_strategy
config.progressive_strategy = args.progressive_strategy
config.proj_strategy = args.proj_strategy
config.contrastive_strategy = args.contrastive_strategy
config.latent_size = args.latent_size
config.stage = args.stage

if args.accelerate:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16)
else:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, config=config, device_map="auto", torch_dtype=torch.bfloat16)


if args.stage in ['stage1-mirage'] or args.stage in ['stage1-progressive']: 
    model.resize_token_embeddings(len(processor.tokenizer))

if args.stage not in ['vanilla-sft']:
    latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
    latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]
    model.config.latent_token_id = int(latent_token_idx)
    model.config.latent_start_id = int(latent_start_idx)
    model.config.latent_end_id = int(latent_end_idx)


for param in model.visual.parameters():
    param.requires_grad = False



def collate_fn_stage1(examples):
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]

    texts = [place_input_image(text) for text in texts]
    texts = [place_output_image(text) for text in texts]
    texts = replace_visual_spectial_tokens(texts)

    image_inputs, _ = process_vision_info(examples)

    user_examples = remove_assistant_images(examples)
    user_text = [processor.apply_chat_template(example, tokenize=False) for example in user_examples]
    user_text = replace_visual_spectial_tokens(user_text)
    user_image_inputs, _ = process_vision_info(user_examples)
    user_batch = processor(text=user_text, images=user_image_inputs, return_tensors="pt", padding=True)

    assistant_examples = remove_user_images(examples)
    assistant_text = [processor.apply_chat_template(example, tokenize=False) for example in assistant_examples]
    assistant_text = replace_visual_spectial_tokens(assistant_text)
    assistant_image_inputs, _ = process_vision_info(assistant_examples)
    assistant_batch = processor(text=assistant_text, images=assistant_image_inputs, return_tensors="pt", padding=True)

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    batch['pixel_values'] = user_batch['pixel_values']
    batch['image_grid_thw'] = user_batch['image_grid_thw']

    batch['pixel_values_latent'] = assistant_batch['pixel_values']
    batch['image_grid_thw_latent'] = assistant_batch['image_grid_thw']

    latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
    latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]

    pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]

    new_input_ids, new_attention_mask = process_batch(batch["input_ids"], batch["attention_mask"], 
                                                      latent_start_idx, latent_end_idx, latent_token_idx, args.latent_size, pad_token_idx)

    batch["input_ids"] = new_input_ids
    batch["attention_mask"] = new_attention_mask

    answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]

    labels = generate_labels_after_multi_token_start(batch["input_ids"], answer_start_token_pattern, pad_token_idx, latent_token_idx)
    batch["labels"] = labels

    image_out_mask = mask_image_output_tokens(batch["input_ids"], latent_start_idx, latent_token_idx)
    batch["image_out_mask"] = image_out_mask  # index <|latent|> embeddings quickly when computing the loss

    return batch

def collate_fn_stage1_progressive(examples):
    end_state_conversations = []
    examples_raw = examples.copy()
    examples = []
    for (example, end_state_conversation) in examples_raw:
        end_state_conversations.append(end_state_conversation)
        examples.append(example)
        
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]

    texts = [place_input_image(text) for text in texts]
    texts = [place_output_image(text) for text in texts]
    texts = replace_visual_spectial_tokens(texts)

    # print(f"example:\n{examples}")
    image_inputs, _ = process_vision_info(examples)
    # print(f"image_inputs:\n{image_inputs}")

    user_examples = remove_assistant_images(examples)  # drop assistant images
    
    user_text = [processor.apply_chat_template(example, tokenize=False) for example in user_examples]
    user_text = replace_visual_spectial_tokens(user_text)
    user_image_inputs, _ = process_vision_info(user_examples)
    user_batch = processor(text=user_text, images=user_image_inputs, return_tensors="pt", padding=True)

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    batch['pixel_values'] = user_batch['pixel_values']
    batch['image_grid_thw'] = user_batch['image_grid_thw']

    batch['start_state'] = user_batch['pixel_values']
    batch['start_state_thw'] = user_batch['image_grid_thw']

    latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
    latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]

    pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]

    new_input_ids, new_attention_mask = process_batch(batch["input_ids"], batch["attention_mask"], 
                                                      latent_start_idx, latent_end_idx, latent_token_idx, args.latent_size, pad_token_idx)

    batch["input_ids"] = new_input_ids
    batch["attention_mask"] = new_attention_mask

    answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]

    labels = generate_labels_after_multi_token_start(batch["input_ids"], answer_start_token_pattern, pad_token_idx, latent_token_idx)
    batch["labels"] = labels

    image_out_mask = mask_image_output_tokens(batch["input_ids"], latent_start_idx, latent_token_idx)  # attention mask that is 1 only on the <latent> span
    batch["image_out_mask"] = image_out_mask  # index <|latent|> embeddings quickly when computing the loss

    inputs_text = [
        processor.tokenizer.decode(ids, skip_special_tokens=False)
        for ids in batch['input_ids']
    ]

    return batch

def collate_fn_stage2(examples):
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    
    texts = [place_input_image(text) for text in texts]
    texts = [place_output_image(text) for text in texts]
    texts = replace_visual_spectial_tokens(texts)
    
    image_inputs, _ = process_vision_info(examples)

    user_examples = remove_assistant_images(examples)
    user_text = [processor.apply_chat_template(example, tokenize=False) for example in user_examples]
    user_text = replace_visual_spectial_tokens(user_text)
    user_image_inputs, _ = process_vision_info(user_examples)
    user_batch = processor(text=user_text, images=user_image_inputs, return_tensors="pt", padding=True)

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    
    batch['pixel_values'] = user_batch['pixel_values']
    batch['image_grid_thw'] = user_batch['image_grid_thw']

    latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
    latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]

    pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]

    new_input_ids, new_attention_mask = process_batch(batch["input_ids"], batch["attention_mask"], 
                                                      latent_start_idx, latent_end_idx, latent_token_idx, args.latent_size, pad_token_idx)

    batch["input_ids"] = new_input_ids
    batch["attention_mask"] = new_attention_mask

    answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]

    labels = generate_labels_after_multi_token_start(batch["input_ids"], answer_start_token_pattern, pad_token_idx, latent_token_idx)
    batch["labels"] = labels
    
    return batch

def collate_fn_stage2_progressive(examples):
    end_state_conversations = []
    examples_raw = examples.copy()
    examples = []
    # for (example, end_state_conversation) in examples_raw:  # original collate_fn_stage2_progressive implementation
    for item in examples_raw:
        if len(item) == 3:
            example, end_state_conversation, third = item
        else:
            example, end_state_conversation = item
        end_state_conversations.append(end_state_conversation)
        examples.append(example)

    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]  # insert <vision_start><image_pad><vision_end> (both input and output contain image data)

    texts = [place_input_image(text) for text in texts]  # before assistant: <image> -> <|vision_start|><|image_pad|><|vision_end|>
    texts = [place_output_image(text) for text in texts]  # if <output_image> exists:  <output_image> -> <|vision_start|><|image_pad|><|vision_end|>
    texts = replace_visual_spectial_tokens(texts)  # after assistant: <|vision_start|><|image_pad|><|vision_end|> -> <|latent_start|><|image_pad|><|latent_end|>, so <|image_pad|> becomes latent_pad

    image_inputs, _ = process_vision_info(examples)
    image_end_states, _ = process_vision_info(end_state_conversations)
    image_end_states_batch = processor(text="", images=image_end_states, return_tensors="pt", padding=True)

    user_examples = remove_assistant_images(examples)
    user_text = [processor.apply_chat_template(example, tokenize=False) for example in user_examples]
    user_text = replace_visual_spectial_tokens(user_text)
    user_image_inputs, _ = process_vision_info(user_examples)
    user_batch = processor(text=user_text, images=user_image_inputs, return_tensors="pt", padding=True)

    assistant_examples = remove_user_images(examples)
    assistant_text = [processor.apply_chat_template(example, tokenize=False) for example in assistant_examples]
    assistant_text = replace_visual_spectial_tokens(assistant_text)
    assistant_image_inputs, _ = process_vision_info(assistant_examples)
    assistant_batch = processor(text=assistant_text, images=assistant_image_inputs, return_tensors="pt", padding=True)

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    batch['pixel_values'] = user_batch['pixel_values']
    batch['image_grid_thw'] = user_batch['image_grid_thw']

    # batch['pixel_values_latent'] = assistant_batch['pixel_values']  # Mirage approach: replace input_embeds with image_embeds; skip this for the progressive method
    # batch['image_grid_thw_latent'] = assistant_batch['image_grid_thw']

    batch['start_state'] = user_batch['pixel_values']
    batch['end_state'] = image_end_states_batch['pixel_values']
    batch['start_state_thw'] = user_batch['image_grid_thw']
    batch['end_state_thw'] = image_end_states_batch['image_grid_thw']

    # print(batch['end_state'] == batch['pixel_values_latent'])  # True: the end state is correct

    latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
    latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]

    pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]

    new_input_ids, new_attention_mask = process_batch(batch["input_ids"], batch["attention_mask"], 
                                                      latent_start_idx, latent_end_idx, latent_token_idx, args.latent_size, pad_token_idx)

    batch["input_ids"] = new_input_ids
    batch["attention_mask"] = new_attention_mask

    answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]

    labels = generate_labels_after_multi_token_start(batch["input_ids"], answer_start_token_pattern, pad_token_idx, latent_token_idx)
    batch["labels"] = labels

    image_out_mask = mask_image_output_tokens(batch["input_ids"], latent_start_idx, latent_token_idx)
    batch["image_out_mask"] = image_out_mask

    inputs_text = [
        processor.tokenizer.decode(ids, skip_special_tokens=False)
        for ids in batch['input_ids']
    ]

    return batch

def collate_fn_vanilla_sft(examples):
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    
    texts = [place_input_image(text) for text in texts]
    
    image_inputs, _ = process_vision_info(examples)

    user_text = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    user_image_inputs, _ = process_vision_info(examples)
    user_batch = processor(text=user_text, images=user_image_inputs, return_tensors="pt", padding=True)

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    
    batch['pixel_values'] = user_batch['pixel_values']
    batch['image_grid_thw'] = user_batch['image_grid_thw']

    pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]

    answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]

    labels = generate_labels_after_multi_token_start(batch["input_ids"], answer_start_token_pattern, pad_token_idx, 151655)  # 151655 is a dummy id. Previously the latent-pad label was set to -100; that is no longer needed

    batch["labels"] = labels
    
    return batch



preprocess_function = task_preporcess_config[args.task]
raw_dataset = load_jsonl_dataset(args.data_path)
train_dataset = [preprocess_function(sample) for sample in raw_dataset]

if args.stage in ['stage1-mirage']:
    CustomTrainer = CustomTrainerStage1
    collate_fn = collate_fn_stage1
elif args.stage in ['stage1-progressive']:
    CustomTrainer = CustomTrainerStage1_progressive
    collate_fn = collate_fn_stage1_progressive
elif args.stage in ['stage2-progressive'] and args.progressive_strategy == "meanPool":
    CustomTrainer = CustomTrainerStage2_progressive_meanPool
    collate_fn = collate_fn_stage2_progressive
elif args.stage in ['vanilla-sft']:
    CustomTrainer = SFTTrainer
    collate_fn = collate_fn_vanilla_sft
else:
    CustomTrainer = CustomTrainerStage2
    collate_fn = collate_fn_stage2

print("=" * 30)
print(f"Task:  {args.task}")
print(f"Stage:  {args.stage}")
print(f"Load model path:  {model_path}")
print(f"Save model path:  {args.save_model_path}")
print(f"Epochs:  {args.epochs}")
print(f"LR:      {args.lr}")
print("=" * 30)

if args.accelerate:
    training_args = SFTConfig(
        output_dir=args.save_model_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # warmup_ratio=0.002,  # 10/5000
        warmup_steps=10,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy="no",  # do not auto-save during training
        optim="adamw_torch_fused",
        bf16=True,
        push_to_hub=False,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to=["wandb"],
        logging_dir='./logs/',
        logging_strategy='steps',
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=True,
    )
else:
    training_args = SFTConfig(
        output_dir=args.save_model_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=10,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=20,
        save_total_limit=1,  # keep only one checkpoint
        save_strategy="no",  # do not auto-save during training
        optim="adamw_torch_fused",
        bf16=True,
        push_to_hub=False,
        remove_unused_columns=False,
        gradient_checkpointing=grad_checkpointing,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to=["wandb"],
        logging_dir='./logs/',
        logging_strategy='steps',
    )


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    tokenizer=processor.tokenizer,
)


trainer.train()
if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

trainer.save_model(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)

if args.accelerate:
    accelerator.end_training()
else:
    wandb.finish()

