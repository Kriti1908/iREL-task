from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="cpu"   # CPU-only as per your constraint
)

messages = [
    {"role": "user", "content": "Give a short introduction to large language models."}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)

inputs = tokenizer([text], return_tensors="pt")

generated_ids = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.6,
    top_p=0.95,
    top_k=20
)

output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()

# Parse thinking vs final answer
try:
    index = len(output_ids) - output_ids[::-1].index(151668)  # </think>
except ValueError:
    index = 0

thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True)
final = tokenizer.decode(output_ids[index:], skip_special_tokens=True)

print("THINKING:\n", thinking)
print("\nANSWER:\n", final)
