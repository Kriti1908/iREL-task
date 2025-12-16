import faiss
import numpy as np
import subprocess
from sentence_transformers import SentenceTransformer
from tools import youtube_search, arxiv_search  # :contentReference[oaicite:5]{index=5}

INDEX_PATH = "vectorstore/index.faiss"
TEXTS_PATH = "vectorstore/texts.npy"
META_PATH = "vectorstore/meta.npy"

LLAMA_BIN = "./llama.cpp/build/bin/llama-cli"
MODEL_PATH = "models/qwen3/qwen3-4b-q4_k_m.gguf"

embedder = SentenceTransformer("intfloat/e5-base-v2")

index = faiss.read_index(INDEX_PATH)
texts = np.load(TEXTS_PATH, allow_pickle=True)
metadata = np.load(META_PATH, allow_pickle=True)

def retrieve_context(question, k=4):
    q_emb = embedder.encode([question]).astype("float32")
    _, idx = index.search(q_emb, k)
    return "\n\n".join(texts[i] for i in idx[0])

def generate_answer(question, thinking=True):
    context = retrieve_context(question)

    prompt = f"""
You are an IIIT course assistant.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

    cmd = [
        LLAMA_BIN,
        "-m", MODEL_PATH,
        "-p", prompt,
        "-n", "512",
        "--temp", "0.6" if thinking else "0.7",
        "--top-p", "0.95" if thinking else "0.8"
    ]

    output = subprocess.check_output(cmd, text=True)

    yt = youtube_search(question)
    arxiv = arxiv_search(question)

    return {
        "answer": output,
        "youtube": yt,
        "papers": arxiv
    }