# fine_tune_gpt_summarization_lora.py

import os
from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# -----------------------------
# 1. Config
# -----------------------------

MODEL_NAME = "gpt2"  # use a small model for testing; swap with a better one later
DATASET_NAME = "cnn_dailymail"
DATASET_CONFIG = "3.0.0"  # cnn_dailymail has 'article' and 'highlights'
MAX_SOURCE_TOKENS = 512
MAX_TARGET_TOKENS = 128

# Simple instruction template
INSTRUCTION_TEMPLATE = """You are a helpful assistant that summarizes news articles.

Article:
{article}

Summary:
"""

# -----------------------------
# 2. Load model & tokenizer
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# gpt2 has no pad_token; set it to eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model (optionally with 8-bit)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True if torch.cuda.is_available() else False,
    device_map="auto" if torch.cuda.is_available() else None,
)

# Prepare for k-bit (8-bit) training if using it
if hasattr(model, "get_input_embeddings") and any(p.dtype == torch.int8 for p in model.parameters()):
    model = prepare_model_for_kbit_training(model)

# -----------------------------
# 3. LoRA config & wrap model
# -----------------------------

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["c_attn", "c_proj"],  # common for GPT2; adjust for other models
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -----------------------------
# 4. Load dataset
# -----------------------------

# cnn_dailymail returns fields: 'article', 'highlights'
raw_datasets = load_dataset(DATASET_NAME, DATASET_CONFIG)

# Use a smaller subset for demo
raw_train = raw_datasets["train"].select(range(2000))
raw_val = raw_datasets["validation"].select(range(500))


# -----------------------------
# 5. Preprocessing: build prompt + tokenize
# -----------------------------

def build_example(example: Dict) -> Dict:
    article = example["article"]
    summary = example["highlights"]

    prompt = INSTRUCTION_TEMPLATE.format(article=article)

    # Full text = prompt + gold summary
    full_text = prompt + summary

    # Tokenize full text
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_SOURCE_TOKENS + MAX_TARGET_TOKENS,
        padding="max_length",
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    # We need labels: ignore loss for prompt tokens, use real IDs for summary tokens
    # Strategy: find where "Summary:\n" ends, and start label there
    # This is approximate but works if template is stable.
    prompt_only = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_SOURCE_TOKENS + MAX_TARGET_TOKENS,
        padding="max_length",
    )["input_ids"]

    # Summary starts at the first index where prompt_only has eos/pad beyond actual prompt length
    # More robust: find exact length before padding
    prompt_len = sum(1 for t in prompt_only if t != tokenizer.pad_token_id)

    labels = [-100] * len(input_ids)
    for i in range(prompt_len, len(input_ids)):
        labels[i] = input_ids[i]

    tokenized["labels"] = labels
    return tokenized


train_dataset = raw_train.map(build_example, batched=False, remove_columns=raw_train.column_names)
val_dataset = raw_val.map(build_example, batched=False, remove_columns=raw_val.column_names)

# -----------------------------
# 6. Data collator
# -----------------------------

@dataclass
class DataCollatorForCausalLM:
    tokenizer: AutoTokenizer
    mlm: bool = False

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Standard LM data collator but we already have labels; just pad.
        batch = {}
        # stack each key
        keys = features[0].keys()
        for key in keys:
            vals = [torch.tensor(f[key]) for f in features]
            batch[key] = torch.stack(vals, dim=0)
        return batch

data_collator = DataCollatorForCausalLM(tokenizer=tokenizer)

# -----------------------------
# 7. TrainingArguments & Trainer
# -----------------------------

training_args = TrainingArguments(
    output_dir="./gpt2-summarization-lora",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    bf16=False,
    report_to="none",
    remove_unused_columns=False,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

trainer.train()

# Save PEFT adapters
trainer.save_model()  # saves the LoRA adapter weights to output_dir
tokenizer.save_pretrained("checkpoints/gpt2-summarization-lora")
