"""
api.py
------
FastAPI REST API for the RAG assistant.

Endpoints:
  POST /ingest          – ingest a file or directory
  POST /query           – ask a question, get an answer + sources
  POST /query/stream    – streaming answer
  DELETE /history       – clear chat history
  GET  /health          – health check
  GET  /docs            – auto-generated Swagger UI (built-in FastAPI)

Run with:
    uvicorn backend.api:app --reload --port 8000
"""

import os
from typing import Optional, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Local modules
from embedding import get_embedding_model, DEFAULT_MODEL as DEFAULT_EMBED_MODEL
from vectordb import load_chroma_vectorstore
from rag_pipeline import RAGPipeline, get_llm
from ingest import ingest as run_ingest


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Assistant API",
    description="Retrieval-Augmented Generation chatbot backend.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state (initialised on startup)
# ---------------------------------------------------------------------------

_pipeline: Optional[RAGPipeline] = None
PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
EMBED_MODEL = os.environ.get("EMBED_MODEL", DEFAULT_EMBED_MODEL)


@app.on_event("startup")
async def startup_event():
    """Load vector store and initialise the RAG pipeline at startup."""
    global _pipeline
    try:
        embeddings = get_embedding_model(EMBED_MODEL)
        vectorstore = load_chroma_vectorstore(embeddings, persist_directory=PERSIST_DIR)
        _pipeline = RAGPipeline(
            vectorstore=vectorstore,
            retrieval_strategy="mmr",
            groq_api_key=GROQ_API_KEY,
        )
        print("[api] ✅ RAG pipeline ready")
    except FileNotFoundError:
        print(
            "[api] ⚠️  No vector store found – POST /ingest first to index documents."
        )


def _get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store not initialised. POST /ingest to index documents first.",
        )
    return _pipeline


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    source: str = Field(..., description="File path, directory, or SQLite .db path.")
    source_type: str = Field("auto", description="auto | text | directory | pdf | sql")
    chunking: str = Field("recursive", description="recursive | character | markdown | semantic | agentic")
    chunk_size: int = Field(500, ge=50, le=4000)
    chunk_overlap: int = Field(50, ge=0, le=500)
    embed_model: str = Field(DEFAULT_EMBED_MODEL)
    incremental: bool = Field(False, description="Add to existing store instead of rebuilding.")


class IngestResponse(BaseModel):
    message: str
    source: str
    chunking: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question.")
    k: int = Field(4, ge=1, le=20, description="Number of chunks to retrieve.")
    strategy: str = Field("mmr", description="mmr | similarity")
    use_history: bool = Field(False, description="Include chat history for follow-ups.")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    chunks: List[str]


class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="ok",
        pipeline_ready=_pipeline is not None,
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_documents(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Index a new data source into the vector store.
    If the store already exists and `incremental=True`, documents are appended.
    """
    global _pipeline

    def _run():
        global _pipeline
        vectorstore = run_ingest(
            source=request.source,
            source_type=request.source_type,
            chunking_strategy=request.chunking,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            embed_model_name=request.embed_model,
            vector_backend="chroma",
            persist_directory=PERSIST_DIR,
            incremental=request.incremental,
        )
        embeddings = get_embedding_model(request.embed_model)
        _pipeline = RAGPipeline(
            vectorstore=vectorstore,
            groq_api_key=GROQ_API_KEY,
        )

    background_tasks.add_task(_run)
    return IngestResponse(
        message="Ingestion started in background. Check /health for readiness.",
        source=request.source,
        chunking=request.chunking,
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """Answer a question using the RAG pipeline."""
    pipeline = _get_pipeline()

    # Temporarily override retrieval settings if caller specified them
    pipeline.retriever = __import__("retriever").get_retriever(
        pipeline.vectorstore,
        strategy=request.strategy,
        k=request.k,
    )

    result = pipeline.query(request.question, use_history=request.use_history)
    return QueryResponse(**result)


@app.post("/query/stream", tags=["Query"])
async def query_stream(request: QueryRequest):
    """
    Stream the LLM answer token by token.
    Returns a text/event-stream response.
    """
    pipeline = _get_pipeline()

    from langchain_core.documents import Document as LCDoc
    from prompt import build_prompt_string
    from langchain_core.messages import HumanMessage

    # Retrieve
    retrieved_docs: List[LCDoc] = pipeline.retriever.invoke(request.question)
    prompt_text = build_prompt_string(retrieved_docs, request.question)

    async def token_generator():
        for chunk in pipeline.llm.stream([HumanMessage(content=prompt_text)]):
            yield chunk.content

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@app.delete("/history", tags=["System"])
async def clear_history():
    """Clear the multi-turn chat history."""
    pipeline = _get_pipeline()
    pipeline.clear_history()
    return {"message": "Chat history cleared."}
