from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3-4B"
LORA_PATH = "lora_out"
OUTPUT_PATH = "qwen3-4b-course"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="cpu",
    torch_dtype="auto"
)

# Attach LoRA adapters
model = PeftModel.from_pretrained(model, LORA_PATH)

# Merge LoRA weights INTO base model
model = model.merge_and_unload()

# Save merged model
model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print("✅ LoRA merged model saved to:", OUTPUT_PATH)