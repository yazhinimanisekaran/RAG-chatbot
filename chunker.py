"""
chunker.py
----------
Text splitting strategies extracted from the notebooks:

  1. CharacterTextSplitter      – simple separator-based splitting
  2. RecursiveCharacterSplitter – hierarchical splitting (recommended default)
  3. MarkdownTextSplitter       – Markdown-aware splitting
  4. SemanticChunker            – embedding-guided semantic boundaries
  5. AgenticChunker             – LLM-based atomic proposition extraction

All chunkers accept a list of Documents and return a list of Documents.
"""

import json
import os
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
)


# ---------------------------------------------------------------------------
# 1. Character Text Splitter
# ---------------------------------------------------------------------------

def chunk_by_character(
    documents: List[Document],
    separator: str = "\n",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    Split documents on a single separator character.
    Simple but doesn't respect paragraph / sentence structure.
    """
    splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


# ---------------------------------------------------------------------------
# 2. Recursive Character Text Splitter  ← recommended default
# ---------------------------------------------------------------------------

def chunk_recursive(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: List[str] | None = None,
) -> List[Document]:
    """
    Hierarchical splitting: tries paragraph → sentence → word → character.
    Best general-purpose chunker for most document types.

    Args:
        documents:     Input documents.
        chunk_size:    Max characters per chunk.
        chunk_overlap: Characters shared between adjacent chunks (preserves context).
        separators:    Override default separator hierarchy if needed.
    """
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(documents)


# ---------------------------------------------------------------------------
# 3. Markdown Text Splitter
# ---------------------------------------------------------------------------

def chunk_markdown(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    Splits Markdown documents at heading boundaries (##, ###, etc.).
    Use when ingesting .md files to keep sections coherent.
    """
    splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


# ---------------------------------------------------------------------------
# 4. Semantic Chunker  (requires embedding model)
# ---------------------------------------------------------------------------

def chunk_semantic(
    documents: List[Document],
    embeddings,
    breakpoint_threshold_type: str = "percentile",
) -> List[Document]:
    """
    Uses embedding similarity to detect topic shifts and split there.
    Produces semantically coherent chunks at the cost of extra embedding calls.

    Args:
        documents:                 Input documents.
        embeddings:                A LangChain Embeddings instance
                                   (e.g. HuggingFaceEmbeddings).
        breakpoint_threshold_type: "percentile" | "standard_deviation" | "interquartile"
    """
    # Lazy import – only needed when this function is called
    from langchain_experimental.text_splitter import SemanticChunker

    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
    )
    return splitter.split_documents(documents)


# ---------------------------------------------------------------------------
# 5. Agentic Chunker  (requires LLM)
# ---------------------------------------------------------------------------

def chunk_agentic(documents: List[Document], llm) -> List[Document]:
    """
    Uses an LLM to extract atomic, self-contained propositions from each chunk.
    Produces the highest-quality chunks but is slow and costs tokens.

    Args:
        documents: Input documents.
        llm:       A LangChain chat model (e.g. ChatGroq).

    Returns:
        One Document per extracted proposition with metadata
        {"chunking_type": "agentic", "source": <original source>}.
    """
    from pydantic import BaseModel, ValidationError
    from langchain_core.prompts import ChatPromptTemplate

    class Propositions(BaseModel):
        propositions: List[str]

    prompt = ChatPromptTemplate.from_template(
        """
You are an expert information extraction agent.

TASK:
Extract ATOMIC propositions from the text.

RULES:
- Each proposition must express exactly ONE factual idea
- No compound sentences
- No explanations
- No markdown
- Output ONLY valid JSON

JSON FORMAT:
{{
  "propositions": ["fact 1", "fact 2"]
}}

TEXT:
{text}
"""
    )

    result_docs: List[Document] = []

    for doc in documents:
        messages = prompt.format_messages(text=doc.page_content)
        response = llm.invoke(messages)

        try:
            content = response.content
            json_start = content.find("{")
            json_end = content.rfind("}")
            if json_start == -1 or json_end == -1 or json_end <= json_start:
                raise ValueError("No valid JSON object found in LLM response.")
            raw_json = json.loads(content[json_start: json_end + 1])
            validated = Propositions(**raw_json)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise ValueError(
                f"Agentic chunking failed for a document.\n"
                f"Error: {exc}\nRaw LLM output:\n{response.content}"
            ) from exc

        for prop in validated.propositions:
            result_docs.append(
                Document(
                    page_content=prop,
                    metadata={
                        "chunking_type": "agentic",
                        "source": doc.metadata.get("source", "unknown"),
                    },
                )
            )

    return result_docs


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

STRATEGY_MAP = {
    "character": chunk_by_character,
    "recursive": chunk_recursive,
    "markdown": chunk_markdown,
    "semantic": chunk_semantic,
    "agentic": chunk_agentic,
}


def chunk_documents(
    documents: List[Document],
    strategy: str = "recursive",
    **kwargs,
) -> List[Document]:
    """
    Unified chunking interface.

    Args:
        documents: Documents returned by parser.load_documents().
        strategy:  One of "recursive" (default), "character", "markdown",
                   "semantic", "agentic".
        **kwargs:  Forwarded to the chosen chunker.
                   - For "semantic": pass embeddings=<HuggingFaceEmbeddings instance>
                   - For "agentic":  pass llm=<ChatGroq instance>

    Returns:
        List of chunked Document objects.
    """
    if strategy not in STRATEGY_MAP:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Valid options: {list(STRATEGY_MAP)}"
        )

    return STRATEGY_MAP[strategy](documents, **kwargs)
