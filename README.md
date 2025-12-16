# IIIT Qwen3 Course Assistant

A CPU-only RAG + LLM system fine-tuned for an IIIT course using Qwen3.

## Features
- Exam question solving
- Step-by-step explanations
- YouTube + research paper suggestions
- Source citation
- No closed APIs

## Setup
```bash
pip install -r requirements.txt
python scripts/parse_pdfs.py
python scripts/build_embeddings.py
python scripts/endsem_eval.py
```

## Model

* Qwen3-4B (INT4)
* RAG + FAISS

---