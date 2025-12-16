import os
import re
import json
import fitz  # PyMuPDF
from tqdm import tqdm

RAW_DIR = "../data/raw/LoRA"
OUT_FILE = "../data/lora_train.jsonl"

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# Regex patterns (case-insensitive, robust)
QUESTION_PATTERN = re.compile(
    r"^\s*(Q(?:uestion)?\s*\d*)\s*[:.]?",
    re.IGNORECASE | re.MULTILINE
)

ANSWER_PATTERN = re.compile(
    r"^\s*(A(?:nswer|ns)?|Sol(?:ution|n|ved)?)\b\s*[:.]?",
    re.IGNORECASE | re.MULTILINE
)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = []
    for page in doc:
        text.append(page.get_text())
    return "\n".join(text)

def clean_text(s):
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def extract_qa_pairs(text):
    qa_pairs = []

    # Split by question markers
    q_splits = QUESTION_PATTERN.split(text)

    for i in range(1, len(q_splits), 2):
        q_marker = q_splits[i]
        q_block = q_splits[i + 1]

        # Split question block into question + answer
        a_split = ANSWER_PATTERN.split(q_block, maxsplit=1)

        if len(a_split) < 3:
            continue  # No valid answer

        question_text = clean_text(a_split[0])
        answer_text = clean_text(a_split[2])

        # Basic quality filters
        if len(question_text) < 10 or len(answer_text) < 30:
            continue

        qa_pairs.append((question_text, answer_text))

    return qa_pairs

def main():
    total = 0

    with open(OUT_FILE, "w", encoding="utf-8") as out:
        for file in tqdm(os.listdir(RAW_DIR)):
            if not file.lower().endswith(".pdf"):
                continue

            pdf_path = os.path.join(RAW_DIR, file)
            text = extract_text_from_pdf(pdf_path)

            qa_pairs = extract_qa_pairs(text)

            for q, a in qa_pairs:
                json.dump(
                    {
                        "instruction": q,
                        "response": a
                    },
                    out,
                    ensure_ascii=False
                )
                out.write("\n")
                total += 1

    print(f"✅ Created {total} LoRA training examples")

if __name__ == "__main__":
    main()