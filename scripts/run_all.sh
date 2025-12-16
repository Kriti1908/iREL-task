#!/usr/bin/env bash
set -e  # exit immediately on error

echo "=========================================="
echo " IIIT Course Assistant – Full Pipeline"
echo " Qwen3-4B + LoRA + RAG (CPU-only)"
echo "=========================================="

############################################
# CONFIGURATION
############################################

VENV_DIR="~/.venv"
MODEL_NAME="Qwen/Qwen3-4B"

RAW_PDF_DIR="../data/raw"
PARSED_DIR="../data/processed"
VECTOR_DIR="../vectorstore"

LORA_DATA="../data/lora_train.jsonl"
LORA_OUT="lora_out"
MERGED_MODEL="qwen3-4b-course"

LLAMA_DIR="../llama.cpp"
GGUF_F16="qwen3-4b-course-f16.gguf"
GGUF_Q4="qwen3-4b-course-q4_k_m.gguf"

############################################
# STEP 0: Environment
############################################

echo "[0/9] Activating virtual environment..."

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment..."
  python3 -m venv $VENV_DIR
fi

source $VENV_DIR/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

############################################
# STEP 1: Parse PDFs
############################################

echo "[1/9] Parsing course PDFs..."
python parse_pdfs.py

############################################
# STEP 2: Build Embeddings
############################################

echo "[2/9] Building vector embeddings..."
python build_embeddings.py

############################################
# STEP 3: Train LoRA adapters
############################################

echo "[3/9] Training LoRA adapters..."

if [ ! -f "$LORA_DATA" ]; then
  echo "ERROR: LoRA training file not found at $LORA_DATA"
  exit 1
fi

python lora_train.py

############################################
# STEP 4: Merge LoRA into base model
############################################

echo "[4/9] Merging LoRA adapters into base model..."
python merge_lora.py

############################################
# STEP 5: Convert merged model to GGUF
############################################

echo "[5/9] Converting merged model to GGUF..."

cd $LLAMA_DIR

python3 convert_hf_to_gguf.py \
  ../$MERGED_MODEL \
  --outfile $GGUF_F16

############################################
# STEP 6: Quantize for CPU inference
############################################

echo "[6/9] Quantizing model (Q4_K_M)..."

./build/bin/llama-quantize \
  $GGUF_F16 \
  $GGUF_Q4 \
  Q4_K_M

cd ..

############################################
# STEP 7: Sanity test llama.cpp inference
############################################

echo "[7/9] Running sanity test with llama.cpp..."

$LLAMA_DIR/build/bin/llama-cli \
  -m $LLAMA_DIR/$GGUF_Q4 \
  -p "Explain gradient descent in simple terms." \
  -n 128 \
  --temp 0.7 \
  --top-p 0.9

############################################
# STEP 8: Run RAG inference
############################################

echo "[8/9] Running RAG inference test..."
python rag_inference.py

############################################
# STEP 9: End-sem evaluation
############################################

# echo "[9/9] Running end-sem evaluation..."
# python endsem_eval.py

############################################
# DONE
############################################

echo "=========================================="
echo " Pipeline completed successfully!"
echo " Results saved in: results/"
echo "=========================================="