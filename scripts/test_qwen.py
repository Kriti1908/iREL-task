from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "Qwen/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,                 # 🔑 KEY LINE
    device_map="auto",                 # handles CPU placement
    trust_remote_code=True
)

messages = [
    {"role": "user", "content": "Explain gradient descent simply."}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False   # IMPORTANT: disable thinking for now
)

inputs = tokenizer(text, return_tensors="pt")

output = model.generate(
    **inputs,
    max_new_tokens=128,     # 🔑 keep small
    temperature=0.7,
    top_p=0.8,
    top_k=20
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
