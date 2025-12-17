import faiss
import numpy as np
import subprocess
from sentence_transformers import SentenceTransformer
from tools import youtube_search, arxiv_search  # :contentReference[oaicite:5]{index=5}

INDEX_PATH = "../vectorstore/index.faiss"
TEXTS_PATH = "../vectorstore/texts.npy"
META_PATH = "../vectorstore/meta.npy"

LLAMA_BIN = "../llama.cpp/build/bin/llama-cli"
MODEL_PATH = "../llama.cpp/models/qwen3/qwen3-4b-course-q4_k_m.gguf"

embedder = SentenceTransformer("intfloat/e5-base-v2")

index = faiss.read_index(INDEX_PATH)
texts = np.load(TEXTS_PATH, allow_pickle=True)
metadata = np.load(META_PATH, allow_pickle=True)

def retrieve_context(question, k=4):
    q_emb = embedder.encode([question]).astype("float32")
    _, idx = index.search(q_emb, k)
    return "\n\n".join(texts[i] for i in idx[0])

import tempfile
import os
import subprocess

def generate_answer(question, thinking=True):
    context = retrieve_context(question)

    prompt = f"""You are an IIIT course assistant.

        Context:
        {context}

        Question:
        {question}

        Answer clearly, completely, and in a structured manner. End your answer with the line:
        FINAL ANSWER END
        Do not write anything after that.
    """

    safe_prompt = prompt.replace('"', '\\"').replace("\n", "\\n")

    cmd = f"""
        echo "" | {LLAMA_BIN} \
        -m {MODEL_PATH} \
        -p "{safe_prompt}" \
        -n 384 \
        --ctx-size 4096 \
        --temp {0.6 if thinking else 0.7} \
        --top-p {0.95 if thinking else 0.8} \
        --log-disable
    """

    result = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    output = result.stdout + result.stderr
    # Cut at explicit marker
    if "FINAL ANSWER END" in output:
        output = output.split("FINAL ANSWER END")[0]
    # Cut at CLI continuation
    if "\n>" in output:
        output = output.split("\n>")[0]
    output = output.strip()

    return {
        "answer": output,
        "youtube": youtube_search(question),
        "papers": arxiv_search(question)
    }