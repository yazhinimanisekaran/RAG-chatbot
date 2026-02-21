"""
vectordb.py
-----------
Vector store creation, persistence, and querying.

Supports two backends from the notebooks:
  - ChromaDB  ← default (persistent, disk-backed)
  - FAISS     ← in-memory, faster for small corpora / MMR retrieval

Both backends expose the same interface so the rest of the pipeline
is agnostic to which one is in use.
"""

import os
from typing import List, Tuple, Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------------------------------------------------------------
# ChromaDB backend
# ---------------------------------------------------------------------------

def create_chroma_vectorstore(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    persist_directory: str = "./chroma_db",
    collection_name: str = "rag_collection",
):
    """
    Build a ChromaDB vector store from a list of document chunks.

    The store is automatically persisted to `persist_directory` so it
    survives restarts.

    Args:
        chunks:            Chunked documents from chunker.py.
        embeddings:        Embedding model from embedding.py.
        persist_directory: Local path for ChromaDB storage.
        collection_name:   Logical name for the Chroma collection.

    Returns:
        A LangChain Chroma vectorstore object.
    """
    from langchain_community.vectorstores import Chroma

    os.makedirs(persist_directory, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    count = vectorstore._collection.count()
    print(f"[vectordb] ChromaDB created – {count} vectors "
          f"persisted to '{persist_directory}'")
    return vectorstore


def load_chroma_vectorstore(
    embeddings: HuggingFaceEmbeddings,
    persist_directory: str = "./chroma_db",
    collection_name: str = "rag_collection",
):
    """
    Load an existing ChromaDB store from disk without re-ingesting documents.

    Raises FileNotFoundError if `persist_directory` does not exist.
    """
    from langchain_community.vectorstores import Chroma

    if not os.path.isdir(persist_directory):
        raise FileNotFoundError(
            f"ChromaDB directory not found: '{persist_directory}'. "
            "Run ingest.py first."
        )

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    count = vectorstore._collection.count()
    print(f"[vectordb] ChromaDB loaded – {count} vectors from '{persist_directory}'")
    return vectorstore


def add_documents_to_chroma(vectorstore, chunks: List[Document]) -> None:
    """Incrementally add new document chunks to an existing Chroma store."""
    vectorstore.add_documents(chunks)
    count = vectorstore._collection.count()
    print(f"[vectordb] Added {len(chunks)} chunks – total: {count} vectors")


# ---------------------------------------------------------------------------
# FAISS backend
# ---------------------------------------------------------------------------

def create_faiss_vectorstore(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    save_path: Optional[str] = None,
):
    """
    Build an in-memory FAISS vector store.

    Args:
        chunks:    Chunked documents.
        embeddings: Embedding model.
        save_path: If provided, save the index to this directory.

    Returns:
        A LangChain FAISS vectorstore object.
    """
    from langchain_community.vectorstores import FAISS

    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"[vectordb] FAISS index created – {len(chunks)} vectors")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        vectorstore.save_local(save_path)
        print(f"[vectordb] FAISS index saved to '{save_path}'")

    return vectorstore


def load_faiss_vectorstore(
    embeddings: HuggingFaceEmbeddings,
    load_path: str,
    allow_dangerous_deserialization: bool = True,
):
    """Load a previously saved FAISS index from disk."""
    from langchain_community.vectorstores import FAISS

    vectorstore = FAISS.load_local(
        load_path,
        embeddings,
        allow_dangerous_deserialization=allow_dangerous_deserialization,
    )
    print(f"[vectordb] FAISS index loaded from '{load_path}'")
    return vectorstore


# ---------------------------------------------------------------------------
# Shared search utilities
# ---------------------------------------------------------------------------

def similarity_search(
    vectorstore,
    query: str,
    k: int = 4,
) -> List[Document]:
    """
    Basic top-k cosine similarity search.

    Args:
        vectorstore: Any LangChain vectorstore (Chroma or FAISS).
        query:       User question string.
        k:           Number of results to return.
    """
    return vectorstore.similarity_search(query, k=k)


def similarity_search_with_scores(
    vectorstore,
    query: str,
    k: int = 4,
) -> List[Tuple[Document, float]]:
    """
    Same as similarity_search but also returns relevance scores.

    Lower distance = more similar for cosine-distance stores.
    """
    return vectorstore.similarity_search_with_score(query, k=k)


def get_mmr_retriever(
    vectorstore,
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
):
    """
    Build a Maximal Marginal Relevance (MMR) retriever.

    MMR balances relevance AND diversity in results – prevents the
    retriever returning near-duplicate chunks.

    Args:
        vectorstore: Chroma or FAISS vectorstore.
        k:           Number of final results.
        fetch_k:     Candidate pool size before MMR re-ranking.
        lambda_mult: 1.0 = pure relevance, 0.0 = pure diversity.
                     0.5 is a good starting point.
    """
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
        },
    )


# ---------------------------------------------------------------------------
# Unified factory
# ---------------------------------------------------------------------------

def get_vectorstore(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    backend: str = "chroma",
    **kwargs,
):
    """
    Create a vector store with the specified backend.

    Args:
        chunks:    Chunked documents.
        embeddings: Embedding model.
        backend:   "chroma" (default) or "faiss".
        **kwargs:  Forwarded to the backend-specific constructor.

    Returns:
        A LangChain vectorstore.
    """
    if backend == "chroma":
        return create_chroma_vectorstore(chunks, embeddings, **kwargs)
    elif backend == "faiss":
        return create_faiss_vectorstore(chunks, embeddings, **kwargs)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Use 'chroma' or 'faiss'.")
