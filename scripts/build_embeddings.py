import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_DIR = "data/processed"
OUT_DIR = "vectorstore"
os.makedirs(OUT_DIR, exist_ok=True)

model = SentenceTransformer("intfloat/e5-base-v2")

texts = []
metadata = []

for f in os.listdir(DATA_DIR):
    if f.endswith(".txt"):
        with open(os.path.join(DATA_DIR, f), "r", encoding="utf-8") as r:
            chunks = r.read().split("\n\n")
            for c in chunks:
                if len(c.strip()) > 50:
                    texts.append(c)
                    metadata.append(f)

embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, f"{OUT_DIR}/index.faiss")
np.save(f"{OUT_DIR}/meta.npy", metadata)
np.save(f"{OUT_DIR}/texts.npy", texts)