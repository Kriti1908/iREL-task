from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

MODEL = "Qwen/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="cpu",
    torch_dtype=torch.float32
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

dataset = load_dataset("json", data_files="../data/lora/train.jsonl")

def tokenize(example):
    text = f"Question: {example['instruction']}\nAnswer: {example['output']}"
    return tokenizer(text, truncation=True, padding="max_length", max_length=512)

dataset = dataset.map(tokenize, remove_columns=["instruction", "output"])

args = TrainingArguments(
    output_dir="lora_out",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    fp16=False
)

trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    args=args
)

trainer.train()