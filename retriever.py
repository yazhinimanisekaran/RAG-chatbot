"""
retriever.py
------------
Retrieval strategies sitting on top of vectordb.py.

Two strategies (from MMR_Implementation.ipynb and VectorDB_Phase.ipynb):
  1. Similarity   – pure cosine-distance top-k
  2. MMR          – Maximal Marginal Relevance (relevance + diversity)

The module also provides a convenience function that returns a
LangChain-compatible retriever object suitable for use in LCEL chains.
"""

from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


# ---------------------------------------------------------------------------
# Similarity retrieval
# ---------------------------------------------------------------------------

def retrieve_by_similarity(
    vectorstore,
    query: str,
    k: int = 4,
) -> List[Document]:
    """
    Return the top-k most similar chunks for `query`.

    Best when you want the single most relevant context window and
    don't mind duplicates.
    """
    docs = vectorstore.similarity_search(query, k=k)
    print(f"[retriever] similarity – retrieved {len(docs)} docs")
    return docs


def retrieve_by_similarity_with_scores(
    vectorstore,
    query: str,
    k: int = 4,
) -> List[Tuple[Document, float]]:
    """
    Same as retrieve_by_similarity but also returns relevance scores.

    Returns:
        List of (Document, score) where lower score = more similar
        (for L2-distance stores) or higher = more similar (cosine).
    """
    results = vectorstore.similarity_search_with_score(query, k=k)
    for doc, score in results:
        src = doc.metadata.get("source", "unknown")
        print(f"  score={score:.4f}  source={src}")
    return results


# ---------------------------------------------------------------------------
# MMR retrieval
# ---------------------------------------------------------------------------

def retrieve_by_mmr(
    vectorstore,
    query: str,
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
) -> List[Document]:
    """
    Maximal Marginal Relevance retrieval.

    Unlike plain similarity search, MMR penalises chunks that are
    too similar to already-selected results, producing a more diverse
    context window.

    Args:
        vectorstore: Any LangChain vectorstore.
        query:       User question.
        k:           Final number of chunks to return.
        fetch_k:     Candidate pool before MMR re-ranking (should be ≥ k).
        lambda_mult: Trade-off between relevance (1.0) and diversity (0.0).
    """
    docs = vectorstore.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )
    print(f"[retriever] MMR – retrieved {len(docs)} diverse docs "
          f"(fetch_k={fetch_k}, λ={lambda_mult})")
    return docs


# ---------------------------------------------------------------------------
# LangChain retriever objects  (for use in LCEL chains)
# ---------------------------------------------------------------------------

def get_similarity_retriever(vectorstore, k: int = 4) -> BaseRetriever:
    """Return a LangChain BaseRetriever for similarity search."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def get_mmr_retriever(
    vectorstore,
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
) -> BaseRetriever:
    """
    Return a LangChain BaseRetriever backed by MMR.

    This object can be used directly in an LCEL chain:
        chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm
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
# Unified entry point
# ---------------------------------------------------------------------------

def retrieve(
    vectorstore,
    query: str,
    strategy: str = "mmr",
    k: int = 4,
    **kwargs,
) -> List[Document]:
    """
    Retrieve relevant documents using the specified strategy.

    Args:
        vectorstore: Chroma or FAISS vectorstore from vectordb.py.
        query:       User question.
        strategy:    "similarity" or "mmr" (default).
        k:           Number of results.
        **kwargs:    Extra args forwarded to the strategy
                     (e.g. fetch_k=20, lambda_mult=0.6 for MMR).

    Returns:
        List of Document objects.
    """
    if strategy == "similarity":
        return retrieve_by_similarity(vectorstore, query, k=k)
    elif strategy == "mmr":
        return retrieve_by_mmr(vectorstore, query, k=k, **kwargs)
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Use 'similarity' or 'mmr'."
        )


def get_retriever(
    vectorstore,
    strategy: str = "mmr",
    k: int = 4,
    **kwargs,
) -> BaseRetriever:
    """
    Return a LangChain-compatible retriever object.

    Use this when building LCEL chains in rag_pipeline.py.

    Args:
        vectorstore: Vector store from vectordb.py.
        strategy:    "similarity" or "mmr".
        k:           Number of results.
        **kwargs:    Extra args for MMR (fetch_k, lambda_mult).
    """
    if strategy == "similarity":
        return get_similarity_retriever(vectorstore, k=k)
    elif strategy == "mmr":
        return get_mmr_retriever(vectorstore, k=k, **kwargs)
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Use 'similarity' or 'mmr'."
        )
