import fitz
import os
from tqdm import tqdm

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"

os.makedirs(OUT_DIR, exist_ok=True)

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

for root, _, files in os.walk(RAW_DIR):
    for f in tqdm(files):
        if f.endswith(".pdf"):
            path = os.path.join(root, f)
            text = extract_text(path)

            out_path = os.path.join(OUT_DIR, f.replace(".pdf", ".txt"))
            with open(out_path, "w", encoding="utf-8") as w:
                w.write(text)