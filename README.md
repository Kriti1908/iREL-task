# 🎓 IIIT Course Assistant: RAG-Enriched LoRA Model

This project implements a sophisticated educational assistant tailored for IIIT course materials. By combining **LoRA (Low-Rank Adaptation)** fine-tuning of the **Qwen3-4B** model with a **RAG (Retrieval-Augmented Generation)** pipeline, the system provides accurate, context-aware answers to complex technical questions, further enriched with external academic resources.

---

## 🏗️ System Architecture

The system follows a hybrid architecture that blends "parametric knowledge" (learned during fine-tuning) with "external knowledge" (retrieved from course documents).

### **1. Data Processing Layer (`parse_pdfs.py`)**

* **Methodology**: Uses `PyMuPDF` (fitz) to iterate through raw course PDFs (slides, textbooks).
* **Design Choice**: Extracted text is cleaned and saved as `.txt` files to simplify the downstream embedding pipeline and avoid repeated PDF parsing overhead.

### **2. Vector Infrastructure (`build_embeddings.py`)**

* **Embedding Model**: `intfloat/e5-base-v2`. This was chosen for its excellent performance in semantic similarity tasks for technical English.
* **Vector DB**: `FAISS` (Facebook AI Similarity Search).
* **Logic**:
* Documents are split into semantic chunks (paragraphs).
* Chunks under 50 characters are filtered out to maintain high-quality retrieval context.
* An `IndexFlatL2` index is built for high-speed similarity searching.



### **3. Model Fine-Tuning (LoRA)**

* **Base Model**: `Qwen/Qwen3-4B`. This model provides a strong balance between reasoning capability and computational efficiency for 4-bit quantization.
* **Quantization**: **4-bit (NF4)** using `bitsandbytes`. This allows fine-tuning on consumer-grade or Kaggle/Colab GPUs (like the Tesla T4).
* **LoRA Config**:
* `r=8`, `lora_alpha=16`.
* Target Modules: `q_proj`, `v_proj`.
* This ensures the model adapts to the specific "tone" and "instruction-following" style of IIIT course queries without forgetting its general knowledge.



### **4. Enrichment & Inference Engine (`irell.ipynb`)**

The core "intelligence" of the system lies in the `generate_answer` workflow:

1. **Concept Extraction**: The model first identifies 2-3 core mathematical/technical concepts from the user's question.
2. **Dual-Context Retrieval**:
* **Internal**: Pulls top-K relevant chunks from the FAISS index.
* **External**: Uses the extracted concepts to query the `arxiv` API and `Youtube` library.


3. **Structured Generation**: The prompt uses a "Chain of Thought" (CoT) approach, asking the model to think briefly before providing a structured response.

---

## 🛠️ Design Choices & Justifications

The design of the **IIIT Course Assistant** is a strategic hybrid of retrieval and fine-tuning. Below is a detailed breakdown of the specific architectural choices and the reasoning behind them.

---

### **1. Data Ingestion & Semantic Filtering**

The system begins with a robust parsing pipeline designed to maximize the "Signal-to-Noise" ratio of retrieved content.

* **Fitz (PyMuPDF) for PDF Extraction**: Instead of generic parsers, `PyMuPDF` was chosen for its ability to handle complex layouts and mathematical symbols common in course slides.
* **Contextual Chunking**: Text is split by double newlines (`\n\n`), targeting natural paragraph breaks which usually represent a single cohesive concept.
* **The 50-Character Heuristic**: Chunks under 50 characters are automatically discarded. This filters out non-informative artifacts like page numbers, slide headers, and image captions that would otherwise "pollute" the vector space.

### **2. Retrieval Strategy: High-Precision Baseline**

The retrieval layer is built to ensure that the model has the exact "Source of Truth" from the course materials.

* **E5-Base-v2 Embeddings**: The `intfloat/e5-base-v2` model was selected because it is specifically trained for **asymmetric retrieval** tasks where a short user query must match a long technical passage.
* **IndexFlatL2 (Brute Force) Choice**: While many RAG systems use approximate search (like HNSW), this system uses `IndexFlatL2`. For a single course's data (typically a few thousand chunks), FlatL2 provides **100% accurate** retrieval with zero latency overhead, avoiding the precision loss inherent in approximate methods.

### **3. Model Fine-Tuning: Parameter-Efficient Adaptation**

To make the general-purpose **Qwen3-4B** behave like a specialized academic assistant, a Low-Rank Adaptation (LoRA) strategy was implemented.

* **4-bit NF4 Quantization**: Using `BitsAndBytes`, the model is loaded in 4-bit precision. This reduces VRAM usage by ~75% while maintaining near-original performance, allowing a 4B parameter model to be fine-tuned on a single T4 GPU.
* **LoRA (Low-Rank Adaptation)**: Only **0.073%** of the parameters (roughly 2.9 million) are trainable. By targeting the attention layers (`q_proj`, `v_proj`), the system adapts the model's communication style and domain jargon without causing "Catastrophic Forgetting" of its base reasoning capabilities.

### **4. Enrichment Engine: Concept-Aware Tool Use**

The system goes beyond simple RAG by performing **Query Transformation** before calling external APIs.

* **Concept Extraction Step**: Before answering, the model is prompted to isolate 2-3 core mathematical concepts from the query.
* **Strategic Tooling**:
* **YouTube Search**: Provides visual learners with relevant pedagogical videos.
* **ArXiv Search**: Connects the course content to current research, fulfilling the "enriched features" requirement.
* **Justification**: Searching for raw questions (e.g., *"What is X?"*) often fails on YouTube; searching for extracted concepts (e.g., *"Discrete Random Variables"*) yields significantly higher-quality supplementary material.


### **5. Summary Table: Design Choice Trade-offs**

| Component | Choice | Justification |
| --- | --- | --- |
| **Model** | Qwen3-4B | Balanced reasoning; small enough for efficient LoRA. |
| **Embeddings** | E5-v2 | Superior for asymmetric retrieval; outperforms BERT-based models. |
| **Quantization** | 4-bit NF4 | 4x reduction in memory; crucial for local/Kaggle hosting. |
| **Index** | FAISS FlatL2 | Exact similarity search; preferred for small-to-medium datasets. |
| **Enrichment** | Dual API (ArXiv/YT) | Contextualizes course theory with real-world research and visuals. |

---

## 🔄 System Flow

1. **Ingestion**: Course PDFs are converted to text and stored.
2. **Indexing**: Text is chunked, embedded, and stored in a FAISS vector store.
3. **Instruction Tuning**: The Qwen3 model is fine-tuned on a synthetic or curated dataset of course Q&A to learn the subject's domain language.
4. **Querying**:
* User asks: *"Explain the MLE for Poisson distribution."*
* **Step A**: RAG finds the exact slide or textbook page describing Poisson MLE.
* **Step B**: Model extracts keywords: "Poisson Distribution", "Maximum Likelihood Estimation".
* **Step C**: System fetches YouTube tutorials and ArXiv papers on these keywords.
* **Step D**: Model synthesizes the internal context into a clear, LaTeX-formatted answer.

![alt text](<Offline Course RAG Pipeline-2025-12-19-061621.png>)

---

## 📈 Performance on Endsem Evaluation

The system was tested against the `endsem_solutions.pdf`. The evaluation results are stored in `endsem_results.json`.

**Key Features Observed in Results:**

* **Mathematical Accuracy**: The model correctly utilizes LaTeX for formulas ().
* **Resource Relevance**: The YouTube suggestions accurately match the difficulty level of the course.
* **Transparency**: The `thoughts` field in the output JSON shows the model's internal reasoning process before arriving at the final answer.

---

## 🚀 How to Run

1. **Parse Data**:
```bash
python scripts/parse_pdfs.py

```


2. **Generate Embeddings**:
```bash
python scripts/build_embeddings.py

```


3. **Inference**:
Open `irell.ipynb` in Kaggle/Colab, ensure the `vectorstore/` and `qwen3-4b-course/` paths are correct, and run the `generate_answer` function.

---

**Developed for the IIIT Fine-Tuning Task.**
*Includes: RAG, LoRA, Concept-Aware External Search, and Automated Endsem Evaluation.*

---