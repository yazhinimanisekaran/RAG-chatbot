"""
rag_pipeline.py
---------------
Assembles the full RAG chain using LangChain Expression Language (LCEL).

Chain flow:
    user query
        ↓
    retriever  (similarity or MMR)
        ↓
    prompt template
        ↓
    LLM  (Groq – fast, free tier available)
        ↓
    StrOutputParser
        ↓
    {"answer": ..., "sources": [...]}

Public API:
    build_rag_chain(vectorstore, ...)  → callable chain
    ask(chain, question)               → dict with answer + sources
    RAGPipeline                        → stateful class with chat history
"""

import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_groq import ChatGroq
from operator import itemgetter

from retriever import get_retriever
from prompt import RAG_PROMPT, CHAT_RAG_PROMPT, CONDENSE_QUESTION_PROMPT, _format_docs


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def get_llm(
    model: str = "qwen/qwen3-32b",
    temperature: float = 0.0,
    groq_api_key: Optional[str] = None,
) -> ChatGroq:
    """
    Return a ChatGroq LLM instance.

    Args:
        model:        Groq model ID. Default is qwen3-32b (high quality + free tier).
        temperature:  0 = deterministic, higher = more creative.
        groq_api_key: API key. Falls back to GROQ_API_KEY env variable.
    """
    api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "Groq API key is required. Set GROQ_API_KEY environment variable "
            "or pass groq_api_key= to get_llm()."
        )
    return ChatGroq(model=model, temperature=temperature, groq_api_key=api_key)


# ---------------------------------------------------------------------------
# Simple single-turn RAG chain
# ---------------------------------------------------------------------------

def build_rag_chain(
    vectorstore,
    retrieval_strategy: str = "mmr",
    k: int = 4,
    model: str = "qwen/qwen3-32b",
    temperature: float = 0.0,
    groq_api_key: Optional[str] = None,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
):
    """
    Build a single-turn LCEL RAG chain.

    Args:
        vectorstore:        Chroma or FAISS store from vectordb.py.
        retrieval_strategy: "mmr" (default) or "similarity".
        k:                  Number of chunks to retrieve.
        model:              Groq model ID.
        temperature:        LLM temperature.
        groq_api_key:       Groq API key (env var fallback).
        fetch_k:            MMR candidate pool size.
        lambda_mult:        MMR diversity weight.

    Returns:
        An LCEL Runnable that accepts {"input": "question"} and returns
        {"answer": str, "sources": list[str]}.
    """
    retriever = get_retriever(
        vectorstore,
        strategy=retrieval_strategy,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )
    llm = get_llm(model=model, temperature=temperature, groq_api_key=groq_api_key)

    # Build LCEL chain
    # Input schema: {"input": <question string>}
    chain = (
        {
            "context": itemgetter("input") | retriever | RunnableLambda(_format_docs),
            "input": itemgetter("input"),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


def ask(chain, question: str) -> dict:
    """
    Query the RAG chain and return a structured response.

    Args:
        chain:    Chain built by build_rag_chain().
        question: User's question string.

    Returns:
        {"answer": str, "sources": list[str]}
        Note: sources are not tracked in this simple chain – use
        RAGPipeline for source attribution.
    """
    answer = chain.invoke({"input": question})
    return {"answer": answer, "sources": []}


# ---------------------------------------------------------------------------
# Stateful pipeline with source tracking and chat history
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Stateful RAG assistant with:
      - Source attribution (shows which document chunks were retrieved)
      - Optional chat history for multi-turn conversations
      - MMR retrieval by default

    Usage:
        pipeline = RAGPipeline(vectorstore, groq_api_key="gsk_...")
        result = pipeline.query("What is machine learning?")
        print(result["answer"])
        print(result["sources"])
    """

    def __init__(
        self,
        vectorstore,
        retrieval_strategy: str = "mmr",
        k: int = 4,
        model: str = "qwen/qwen3-32b",
        temperature: float = 0.0,
        groq_api_key: Optional[str] = None,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ):
        self.vectorstore = vectorstore
        self.retrieval_strategy = retrieval_strategy
        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.chat_history: List = []

        self.retriever = get_retriever(
            vectorstore,
            strategy=retrieval_strategy,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )
        self.llm = get_llm(
            model=model,
            temperature=temperature,
            groq_api_key=groq_api_key,
        )

    def query(self, question: str, use_history: bool = False) -> dict:
        """
        Answer a question using the RAG pipeline.

        Args:
            question:    User question.
            use_history: If True, prepend chat history to provide context
                         for follow-up questions.

        Returns:
            {
                "question":   str,
                "answer":     str,
                "sources":    list[str],    # unique source file paths
                "chunks":     list[str],    # raw retrieved chunk content
            }
        """
        # Retrieve relevant chunks
        retrieved_docs: List[Document] = self.retriever.invoke(question)

        # Build prompt string
        from prompt import build_prompt_string
        prompt_text = build_prompt_string(retrieved_docs, question)

        # Call LLM
        from langchain_core.messages import HumanMessage
        response = self.llm.invoke([HumanMessage(content=prompt_text)])
        answer = response.content

        # Extract source metadata
        sources = list({
            doc.metadata.get("source", "unknown") for doc in retrieved_docs
        })
        chunks = [doc.page_content for doc in retrieved_docs]

        # Update chat history
        if use_history:
            from langchain_core.messages import AIMessage
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=answer))

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "chunks": chunks,
        }

    def clear_history(self) -> None:
        """Reset the conversation history."""
        self.chat_history = []
        print("[pipeline] Chat history cleared.")

    def add_documents(self, new_docs: List[Document]) -> None:
        """
        Incrementally add new documents to the vector store at runtime.

        Args:
            new_docs: Already-chunked Document objects.
        """
        self.vectorstore.add_documents(new_docs)
        print(f"[pipeline] Added {len(new_docs)} documents to the vector store.")
