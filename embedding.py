"""
embedding.py
------------
Embedding model initialization and similarity utilities.

Supported models (all open-source, no API key required):
  - sentence-transformers/all-MiniLM-L6-v2        ← default, fastest
  - sentence-transformers/all-MiniLM-L12-v2       ← slightly better quality
  - sentence-transformers/all-mpnet-base-v2        ← best quality, slower
  - sentence-transformers/multi-qa-MiniLM-L6-cos-v1 ← optimised for Q&A
  - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 ← multilingual
"""

from typing import List
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------------------------------------------------------------
# Model catalogue  (from Embedding_Phase.ipynb)
# ---------------------------------------------------------------------------

AVAILABLE_MODELS = {
    "all-MiniLM-L6-v2": {
        "full_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimensions": 384,
        "description": "Fast and efficient, good quality",
        "use_case": "General purpose, real-time applications",
    },
    "all-MiniLM-L12-v2": {
        "full_name": "sentence-transformers/all-MiniLM-L12-v2",
        "dimensions": 384,
        "description": "Slightly better than L6, bit slower",
        "use_case": "Good balance of speed and quality",
    },
    "all-mpnet-base-v2": {
        "full_name": "sentence-transformers/all-mpnet-base-v2",
        "dimensions": 768,
        "description": "Best quality, slower than MiniLM",
        "use_case": "When quality matters more than speed",
    },
    "multi-qa-MiniLM-L6-cos-v1": {
        "full_name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "dimensions": 384,
        "description": "Optimized for question-answering",
        "use_case": "Q&A systems, semantic search",
    },
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "full_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dimensions": 384,
        "description": "Supports 50+ languages",
        "use_case": "Multilingual applications",
    },
}

DEFAULT_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def get_embedding_model(
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
) -> HuggingFaceEmbeddings:
    """
    Load and return a HuggingFaceEmbeddings instance.

    Args:
        model_name: Short name (key in AVAILABLE_MODELS) or a full
                    HuggingFace model string like "sentence-transformers/...".
        device:     "cpu" or "cuda".

    Returns:
        A LangChain HuggingFaceEmbeddings object ready for use in
        vectorstores, semantic chunking, etc.
    """
    # Resolve short name to full model path
    if model_name in AVAILABLE_MODELS:
        full_name = AVAILABLE_MODELS[model_name]["full_name"]
    else:
        full_name = model_name  # assume caller passed the full path

    print(f"[embedding] Loading model: {full_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=full_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def embed_query(text: str, embeddings: HuggingFaceEmbeddings) -> List[float]:
    """Embed a single query string."""
    return embeddings.embed_query(text)


def embed_documents(texts: List[str], embeddings: HuggingFaceEmbeddings) -> List[List[float]]:
    """Batch-embed a list of text strings."""
    return embeddings.embed_documents(texts)


# ---------------------------------------------------------------------------
# Cosine similarity  (from Embedding_Phase.ipynb)
# ---------------------------------------------------------------------------

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Cosine similarity between two vectors.

    Returns a value in [-1, 1]:
      ~  1 → very similar
      ~  0 → unrelated
      ~ -1 → opposite
    """
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def find_most_similar(
    query: str,
    candidates: List[str],
    embeddings: HuggingFaceEmbeddings,
) -> List[tuple[str, float]]:
    """
    Rank a list of candidate strings by similarity to the query.

    Returns:
        List of (text, score) tuples sorted descending by score.
    """
    q_vec = embed_query(query, embeddings)
    c_vecs = embed_documents(candidates, embeddings)
    scored = [
        (text, cosine_similarity(q_vec, vec))
        for text, vec in zip(candidates, c_vecs)
    ]
    return sorted(scored, key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def list_available_models() -> None:
    """Print a formatted table of all available embedding models."""
    print(f"\n{'Model':<50} {'Dims':>6}  {'Use-case'}")
    print("-" * 90)
    for short, info in AVAILABLE_MODELS.items():
        print(f"{info['full_name']:<50} {info['dimensions']:>6}  {info['use_case']}")
