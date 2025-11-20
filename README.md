# RAG Demo

A Retrieval-Augmented Generation (RAG) system that demonstrates how to build a question-answering system using document embeddings and vector search.

## Overview

This project implements a complete RAG pipeline:
1. Load and split documents into chunks
2. Generate embeddings using sentence transformers
3. Store embeddings in a FAISS vector index
4. Retrieve relevant context for queries
5. Generate answers using either OpenAI API or a local LLM

## Project Structure

```
rag-demo/
├── data/                      # Source documents
│   ├── supervised_learning.txt
│   └── unsupervised_learning.txt
├── load_data.py              # Load documents from data/ directory
├── split_text.py             # Split documents into chunks
├── create_embeddings.py      # Generate embeddings from text chunks
├── store_faiss.py            # Build and save FAISS index
├── retrieve_faiss.py         # Load index and retrieve similar chunks
├── generate_answer.py        # Generate answers using local LLM (resource-intensive)
├── generate_answer_api.py    # Generate answers using OpenAI API (recommended)
├── main.py                   # Main entry point
├── requirements.txt          # Core dependencies (for API-based LLM)
└── requirements_llm.txt      # Additional dependencies for local LLM
```

## Installation

### Option 1: API-Based LLM (Recommended)
Lighter and faster, uses OpenAI API for answer generation.

```bash
pip install -r requirements.txt
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

### Option 2: Local LLM
Resource-intensive, runs models locally. Requires GPU for reasonable performance.

```bash
pip install -r requirements.txt -r requirements_llm.txt
```

## Usage

### Quick Start

Run the complete pipeline:
```bash
python main.py
```

### Individual Steps

1. **Load documents:**
   ```python
   from load_data import load_documents
   documents = load_documents('data/')
   ```

2. **Split into chunks:**
   ```python
   from split_text import split_docs
   chunks = split_docs(documents, chunk_size=500, chunk_overlap=100)
   ```

3. **Create embeddings:**
   ```python
   from create_embeddings import get_embeddings
   embeddings = get_embeddings([chunk.page_content for chunk in chunks])
   ```

4. **Build FAISS index:**
   ```python
   from store_faiss import build_faiss_index
   index = build_faiss_index(embeddings, save_path="faiss_index")
   ```

5. **Generate answers:**
   
   **Using OpenAI API (recommended):**
   ```python
   from generate_answer_api import generate_answer_api
   generate_answer_api("What is supervised learning?", top_k=3)
   ```
   
   **Using local LLM:**
   ```python
   from generate_answer import generate_answer
   generate_answer("What is supervised learning?", top_k=3)
   ```

## Key Dependencies

### Core (requirements.txt)
- `sentence-transformers` - Generate embeddings
- `faiss-cpu` - Vector similarity search
- `openai` - OpenAI API client
- `scikit-learn` - ML utilities
- `numpy` - Numerical operations

### Local LLM (requirements_llm.txt)
- `torch` - PyTorch framework
- `transformers` - HuggingFace transformers
- `accelerate` - Model loading optimization
- NVIDIA CUDA packages - GPU acceleration

## How It Works

1. **Document Loading**: Reads text files from the `data/` directory
2. **Text Splitting**: Breaks documents into overlapping chunks for better context retention
3. **Embedding Generation**: Converts text chunks to dense vectors using `sentence-transformers/all-MiniLM-L6-v2`
4. **Vector Storage**: Stores embeddings in a FAISS index for fast similarity search
5. **Retrieval**: Finds the most relevant chunks for a given query
6. **Answer Generation**: Uses retrieved context to generate accurate answers via:
   - OpenAI API (gpt-5-nano or other models) - Fast, cloud-based
   - Local LLM (TinyLlama) - Runs on your hardware

## Configuration

- **Chunk size**: Default 500 characters with 100 character overlap
- **Embeddings model**: `sentence-transformers/all-MiniLM-L6-v2`
- **FAISS index**: Flat L2 index for exact nearest neighbor search
- **Top-k retrieval**: Default 3 most similar chunks
- **OpenAI model**: `gpt-5-nano` (configurable in `generate_answer_api.py`)
- **Local LLM model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (configurable in `generate_answer.py`)

## Notes

- The RAG process (retrieval) is identical for both API and local LLM approaches
- API-based approach is recommended for most users (faster, less resource-intensive)
- Local LLM requires significant computational resources (GPU recommended)
- Generated embeddings and FAISS index are saved to disk for reuse
