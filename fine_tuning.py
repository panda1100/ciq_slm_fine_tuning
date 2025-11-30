from unsloth import FastLanguageModel
import torch
from datasets import load_from_disk
from unsloth import to_sharegpt
from unsloth import standardize_sharegpt
from unsloth import apply_chat_template
from trl import SFTConfig, SFTTrainer

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
#model, tokenizer = FastLanguageModel.from_pretrained(
#    model_name = "model",
#    max_seq_length = max_seq_length,
#    dtype = dtype,
#    load_in_4bit = load_in_4bit,
#    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
#)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

dataset = load_from_disk("/scratch/tokenized_dataset")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = False, # Can make training 5x faster for short sequences.
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        # num_train_epochs = 1, # For longer training runs!
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use TrackIO/WandB etc
    ),
)

trainer_stats = trainer.train()

import traceback

try:
    model.save_pretrained_gguf(
        "model",
        tokenizer,
        quantization_method = "f16"
    )
except Exception:
    print("save_pretrained_gguf FAILED:")
    traceback.print_exc()

try:
    model.save_pretrained_merged(
        "finetuned",
        tokenizer,
        save_method="merged_16bit",
    )
except Exception:
    print("save_pretrained_merged FAILED:")
    traceback.print_exc()

#from pathlib import Path
#import os
#cwd_dir = Path(os.getcwd())
#out_dir = Path("finetuned")
#print("CWD:", os.getcwd())
#print("Exists?", out_dir.exists())
#print("Contents:", list(cwd_dir.iterdir()))
#print("Contents:", list(Path("/scratch/model").iterdir()))
#if out_dir.exists():
#    print("Contents:", list(out_dir.iterdir()))

#import json
#with open("finetuned/config.json", "r", encoding="utf-8") as f:
#    config = json.load(f)
#print(json.dumps(config, indent=2, ensure_ascii=False))


#model.save_pretrained_gguf(
#    "model",
#    tokenizer,
#    #quantization_method = "q4_k_m",  # or "q8_0" / "f16"
#)