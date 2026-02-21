"""
ingest.py
---------
End-to-end ingestion pipeline.

Orchestrates: parser → chunker → embedding → vectordb

Run directly to index documents:
    python backend/ingest.py --source data/ --source_type directory

Or import and call ingest() programmatically from other modules.
"""

import argparse
import os
from typing import List, Optional

from langchain_core.documents import Document

from parser import load_documents
from chunker import chunk_documents
from embedding import get_embedding_model
from vectordb import get_vectorstore, load_chroma_vectorstore, add_documents_to_chroma


# ---------------------------------------------------------------------------
# Config defaults  –  edit here or pass via CLI / env vars
# ---------------------------------------------------------------------------

DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_PERSIST_DIR = "./chroma_db"
DEFAULT_BACKEND = "chroma"
DEFAULT_RETRIEVAL_STRATEGY = "mmr"


# ---------------------------------------------------------------------------
# Core ingestion function
# ---------------------------------------------------------------------------

def ingest(
    source: str,
    source_type: str = "auto",
    chunking_strategy: str = "recursive",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embed_model_name: str = DEFAULT_EMBED_MODEL,
    vector_backend: str = DEFAULT_BACKEND,
    persist_directory: str = DEFAULT_PERSIST_DIR,
    incremental: bool = False,
) -> object:
    """
    Parse → chunk → embed → store a data source.

    Args:
        source:             File path, directory, or DB path.
        source_type:        "auto", "text", "directory", "pdf", or "sql".
        chunking_strategy:  "recursive" | "character" | "markdown" |
                            "semantic" | "agentic".
        chunk_size:         Max characters per chunk.
        chunk_overlap:      Characters overlapping adjacent chunks.
        embed_model_name:   Short name or full HuggingFace model ID.
        vector_backend:     "chroma" (persistent) or "faiss" (in-memory).
        persist_directory:  Where ChromaDB stores its files.
        incremental:        If True and a Chroma store already exists,
                            add to it instead of rebuilding.

    Returns:
        The populated vectorstore object.
    """

    # ---- 1. Load documents ------------------------------------------------
    print(f"\n[ingest] 📂 Loading documents from '{source}' (type={source_type})")
    documents: List[Document] = load_documents(source, source_type=source_type)
    print(f"[ingest] ✅ Loaded {len(documents)} document(s)")

    # ---- 2. Chunk ---------------------------------------------------------
    print(f"\n[ingest] ✂️  Chunking with strategy='{chunking_strategy}' "
          f"(size={chunk_size}, overlap={chunk_overlap})")

    chunking_kwargs: dict = {}

    # Semantic / agentic need extra dependencies loaded up front
    if chunking_strategy == "semantic":
        embeddings_for_chunking = get_embedding_model(embed_model_name)
        chunking_kwargs["embeddings"] = embeddings_for_chunking
    elif chunking_strategy == "agentic":
        groq_api_key = os.environ.get("GROQ_API_KEY", "")
        if not groq_api_key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable is required for agentic chunking."
            )
        from langchain_groq import ChatGroq
        chunking_kwargs["llm"] = ChatGroq(
            model="qwen/qwen3-32b",
            temperature=0,
            groq_api_key=groq_api_key,
        )
    else:
        chunking_kwargs["chunk_size"] = chunk_size
        chunking_kwargs["chunk_overlap"] = chunk_overlap

    chunks: List[Document] = chunk_documents(
        documents, strategy=chunking_strategy, **chunking_kwargs
    )
    print(f"[ingest] ✅ Created {len(chunks)} chunk(s)")

    # ---- 3. Embeddings ----------------------------------------------------
    print(f"\n[ingest] 🔢 Loading embedding model: {embed_model_name}")
    embeddings = get_embedding_model(embed_model_name)

    # ---- 4. Vector store --------------------------------------------------
    if incremental and vector_backend == "chroma" and os.path.isdir(persist_directory):
        print(f"\n[ingest] ➕ Incremental mode – adding to existing ChromaDB")
        vectorstore = load_chroma_vectorstore(embeddings, persist_directory)
        add_documents_to_chroma(vectorstore, chunks)
    else:
        print(f"\n[ingest] 🗄️  Building {vector_backend} vector store")
        vectorstore = get_vectorstore(
            chunks,
            embeddings,
            backend=vector_backend,
            persist_directory=persist_directory,
        )

    print("\n[ingest] 🎉 Ingestion complete!\n")
    return vectorstore


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG vector store."
    )
    parser.add_argument(
        "--source", required=True,
        help="File path, directory, or SQLite .db path to ingest."
    )
    parser.add_argument(
        "--source_type", default="auto",
        choices=["auto", "text", "directory", "pdf", "sql"],
        help="How to interpret the source (default: auto-detect)."
    )
    parser.add_argument(
        "--chunking", default="recursive",
        choices=["recursive", "character", "markdown", "semantic", "agentic"],
        help="Chunking strategy (default: recursive)."
    )
    parser.add_argument(
        "--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Max chunk size in characters (default: {DEFAULT_CHUNK_SIZE})."
    )
    parser.add_argument(
        "--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
        help=f"Chunk overlap in characters (default: {DEFAULT_CHUNK_OVERLAP})."
    )
    parser.add_argument(
        "--embed_model", default=DEFAULT_EMBED_MODEL,
        help=f"Embedding model name (default: {DEFAULT_EMBED_MODEL})."
    )
    parser.add_argument(
        "--backend", default=DEFAULT_BACKEND,
        choices=["chroma", "faiss"],
        help=f"Vector store backend (default: {DEFAULT_BACKEND})."
    )
    parser.add_argument(
        "--persist_dir", default=DEFAULT_PERSIST_DIR,
        help=f"ChromaDB persistence directory (default: {DEFAULT_PERSIST_DIR})."
    )
    parser.add_argument(
        "--incremental", action="store_true",
        help="Add to an existing vector store instead of rebuilding."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ingest(
        source=args.source,
        source_type=args.source_type,
        chunking_strategy=args.chunking,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embed_model_name=args.embed_model,
        vector_backend=args.backend,
        persist_directory=args.persist_dir,
        incremental=args.incremental,
    )
