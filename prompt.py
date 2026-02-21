"""
prompt.py
---------
Prompt templates for the RAG pipeline.

All templates are LangChain PromptTemplate / ChatPromptTemplate objects
so they slot directly into LCEL chains.

Templates provided:
  - RAG_PROMPT             ← default Q&A with cited sources
  - CONDENSE_QUESTION_PROMPT ← rewrite a follow-up question stand-alone
                               (for multi-turn chat history)
  - SYSTEM_PROMPT          ← base system persona for the assistant
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _format_docs(docs) -> str:
    """
    Convert a list of LangChain Documents into a plain-text context block.
    Each chunk is labelled with its source file.
    """
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )


# ---------------------------------------------------------------------------
# 1.  Default RAG prompt  (from VectorDB_Phase.ipynb + MMR_Implementation.ipynb)
# ---------------------------------------------------------------------------

RAG_SYSTEM_MESSAGE = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context. If the answer cannot be found in the context, "
    "reply with \"I don't know based on the provided documents.\"\n\n"
    "Context:\n{context}"
)

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_MESSAGE),
        ("human", "{input}"),
    ]
)


# ---------------------------------------------------------------------------
# 2.  Condense / question-rewriting prompt  (multi-turn chat)
# ---------------------------------------------------------------------------
#  Used to collapse a follow-up question + chat history into a single
#  stand-alone query before retrieval.

CONDENSE_QUESTION_TEMPLATE = (
    "Given the conversation history below and a follow-up question, "
    "rewrite the follow-up question as a self-contained question that "
    "includes all necessary context from the history.\n\n"
    "Chat history:\n{chat_history}\n\n"
    "Follow-up question: {input}\n\n"
    "Self-contained question:"
)

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)


# ---------------------------------------------------------------------------
# 3.  Multi-turn chat RAG prompt  (includes chat history placeholder)
# ---------------------------------------------------------------------------

CHAT_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a knowledgeable assistant. "
            "Answer the user's question using ONLY the context below. "
            "If the answer is not in the context, say \"I don't know.\"\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)


# ---------------------------------------------------------------------------
# 4.  Simple custom prompt builder  (from VectorDB_Phase.ipynb)
# ---------------------------------------------------------------------------

def build_prompt_string(context_docs, question: str) -> str:
    """
    Build a raw prompt string (not a template object).
    Useful when calling the LLM directly without LCEL.

    Args:
        context_docs: List of LangChain Document objects.
        question:     The user's question.

    Returns:
        Formatted prompt string.
    """
    context_text = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in context_docs
    )

    return (
        "You are a helpful assistant.\n"
        "Answer the question using ONLY the provided context.\n"
        "If the answer is not in the context, say \"I don't know\".\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    ).strip()


# ---------------------------------------------------------------------------
# 5.  Prompt selector  (convenience factory)
# ---------------------------------------------------------------------------

PROMPT_REGISTRY = {
    "rag": RAG_PROMPT,
    "condense": CONDENSE_QUESTION_PROMPT,
    "chat_rag": CHAT_RAG_PROMPT,
}


def get_prompt(name: str = "rag"):
    """
    Return a named prompt template.

    Args:
        name: "rag" (default), "condense", or "chat_rag".
    """
    if name not in PROMPT_REGISTRY:
        raise ValueError(
            f"Unknown prompt '{name}'. Available: {list(PROMPT_REGISTRY)}"
        )
    return PROMPT_REGISTRY[name]
