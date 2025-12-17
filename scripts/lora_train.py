from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

MODEL = "Qwen/Qwen3-4B"

# 1️⃣ Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)

# 2️⃣ INT8 base model (CRITICAL)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_config,
    device_map="cpu"
)

# 3️⃣ LoRA config (keep small)
lora_config = LoraConfig(
    r=2,                      # 🔥 critical
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
# model.train()
model.print_trainable_parameters()

# 4️⃣ Dataset
dataset = load_dataset("json", data_files="../data/lora_train.jsonl")

def tokenize(example):
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=64,     # reduced from 256
        padding=False
    )

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(tokenize, remove_columns=["instruction", "response"])

# 5️⃣ Training args (LOW MEMORY)
args = TrainingArguments(
    output_dir="lora_out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,        # 🔥 only 1 epoch
    max_steps=20,              # 🔥 hard cap
    max_grad_norm=0.3,
    optim="adamw_torch",
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    # dataloader_pin_memory=False  # ✅ ADD THIS
)

trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    args=args
)

trainer.train()

model.save_pretrained("lora_out")
tokenizer.save_pretrained("lora_out")