from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "Qwen/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="cpu"
)

def generate_answer(context, question, thinking=True):
    messages = [
        {
            "role": "user",
            "content": f"""
You are an IIIT course assistant.

Context:
{context}

Question:
{question}

Answer step by step and clearly.
"""
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking
    )

    inputs = tokenizer([text], return_tensors="pt")

    output = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.6 if thinking else 0.7,
        top_p=0.95 if thinking else 0.8,
        top_k=20
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded


# Thinking vs non-thinking mode in YOUR system (feature gold)

# You can now expose this as a feature:
# answer = generate_answer(context, question, thinking=True)
# or
# answer = generate_answer(context, question, thinking=False)

# In PPT, say:
# “The system dynamically switches between reasoning-intensive and efficient inference modes using Qwen3’s native thinking switch.”
# That is literally one of Qwen3’s headline features.