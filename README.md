# RAG Assistant

A production-ready Retrieval-Augmented Generation chatbot built with LangChain, ChromaDB, Groq, and Streamlit.

## Project Structure

```
rag-assistant/
├── backend/
│   ├── parser.py          # Load text, PDF, SQL → Documents
│   ├── chunker.py         # Split docs (recursive, semantic, agentic, …)
│   ├── embedding.py       # HuggingFace embedding models + cosine similarity
│   ├── vectordb.py        # ChromaDB & FAISS vector stores
│   ├── retriever.py       # Similarity & MMR retrieval
│   ├── ingest.py          # End-to-end ingestion pipeline (CLI + importable)
│   ├── prompt.py          # Prompt templates (RAG, chat, condense)
│   ├── rag_pipeline.py    # Full LCEL chain + stateful RAGPipeline class
│   └── api.py             # FastAPI REST API
├── frontend/
│   └── app.py             # Streamlit chat UI
└── requirements.txt
```

## Setup

```bash
# 1. Activate your virtual environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Groq API key (get a free key at console.groq.com)
# Windows:
set GROQ_API_KEY=gsk_your_key_here
# Mac/Linux:
export GROQ_API_KEY=gsk_your_key_here
```

## Quickstart

### Step 1 – Ingest documents

```bash
# Index a directory of text files
python backend/ingest.py --source data/ --source_type directory

# Index a single PDF
python backend/ingest.py --source data/report.pdf

# Index a SQLite database
python backend/ingest.py --source data/company.db --source_type sql

# Use semantic chunking instead of recursive
python backend/ingest.py --source data/ --chunking semantic

# Add more docs to an existing store (don't rebuild)
python backend/ingest.py --source data/new_docs/ --incremental
```

### Step 2 – Start the API

```bash
uvicorn backend.api:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

### Step 3 – Launch the UI

```bash
streamlit run frontend/app.py
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check backend status |
| POST | `/ingest` | Index documents |
| POST | `/query` | Ask a question |
| POST | `/query/stream` | Streaming answer |
| DELETE | `/history` | Clear chat history |

### Query example (curl)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?", "k": 4, "strategy": "mmr"}'
```

## Chunking Strategies

| Strategy | Best for | Speed |
|----------|----------|-------|
| `recursive` | General documents (default) | Fast |
| `character` | Simple, uniform text | Fast |
| `markdown` | Markdown / structured docs | Fast |
| `semantic` | Topic-dense documents | Slow (runs embeddings) |
| `agentic` | Maximum chunk quality | Slowest (calls LLM) |

## Retrieval Strategies

| Strategy | Description |
|----------|-------------|
| `mmr` | Maximal Marginal Relevance – balances relevance + diversity (default) |
| `similarity` | Pure cosine-distance top-k |

## Embedding Models

| Short name | Dims | Notes |
|------------|------|-------|
| `all-MiniLM-L6-v2` | 384 | Default, fastest |
| `all-MiniLM-L12-v2` | 384 | Slightly better quality |
| `all-mpnet-base-v2` | 768 | Best quality, slower |
| `multi-qa-MiniLM-L6-cos-v1` | 384 | Optimised for Q&A |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Multilingual |
