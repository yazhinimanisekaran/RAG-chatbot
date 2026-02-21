"""
frontend/app.py
---------------
Streamlit chat interface for the RAG assistant.

Run with:
    streamlit run frontend/app.py

Requires the FastAPI backend to be running:
    uvicorn backend.api:app --reload --port 8000

Features:
  - Chat-style Q&A interface
  - Source attribution (expandable)
  - Ingest new documents via sidebar
  - Retrieval strategy selector (MMR / Similarity)
  - Chat history toggle
  - Clear history button
"""

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {"role": "user"|"assistant", "content": str}

if "use_history" not in st.session_state:
    st.session_state.use_history = False


# ---------------------------------------------------------------------------
# Helper: call API
# ---------------------------------------------------------------------------

def check_health() -> dict:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json()
    except Exception:
        return {"status": "unreachable", "pipeline_ready": False}


def query_api(question: str, k: int, strategy: str, use_history: bool) -> dict:
    payload = {
        "question": question,
        "k": k,
        "strategy": strategy,
        "use_history": use_history,
    }
    r = requests.post(f"{API_BASE}/query", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def ingest_api(source: str, source_type: str, chunking: str,
               chunk_size: int, chunk_overlap: int, incremental: bool) -> dict:
    payload = {
        "source": source,
        "source_type": source_type,
        "chunking": chunking,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "incremental": incremental,
    }
    r = requests.post(f"{API_BASE}/ingest", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def clear_history_api() -> dict:
    r = requests.delete(f"{API_BASE}/history", timeout=5)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Settings")

    # Health indicator
    health = check_health()
    if health["pipeline_ready"]:
        st.success("✅ Backend ready")
    elif health["status"] == "unreachable":
        st.error("❌ Backend unreachable – start the API server")
    else:
        st.warning("⚠️ Vector store not initialised – ingest documents first")

    st.divider()

    # Retrieval settings
    st.subheader("Retrieval")
    strategy = st.selectbox(
        "Strategy",
        options=["mmr", "similarity"],
        help="MMR returns diverse results; similarity returns top-k closest.",
    )
    k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=4)
    use_history = st.toggle(
        "Multi-turn chat history",
        value=st.session_state.use_history,
        help="Send previous Q&A pairs to the LLM for follow-up context.",
    )
    st.session_state.use_history = use_history

    st.divider()

    # Ingest panel
    st.subheader("📂 Ingest Documents")
    with st.expander("Ingest settings", expanded=False):
        ingest_source = st.text_input(
            "Source path",
            placeholder="e.g. data/ or data/report.pdf",
        )
        ingest_type = st.selectbox(
            "Source type",
            ["auto", "directory", "text", "pdf", "sql"],
        )
        ingest_chunking = st.selectbox(
            "Chunking strategy",
            ["recursive", "character", "markdown", "semantic", "agentic"],
        )
        ingest_chunk_size = st.number_input(
            "Chunk size", min_value=100, max_value=4000, value=500, step=50
        )
        ingest_chunk_overlap = st.number_input(
            "Chunk overlap", min_value=0, max_value=500, value=50, step=10
        )
        ingest_incremental = st.checkbox(
            "Incremental (add to existing store)",
            value=False,
        )

        if st.button("🚀 Start Ingest", use_container_width=True):
            if not ingest_source:
                st.error("Please enter a source path.")
            else:
                with st.spinner("Ingestion started…"):
                    try:
                        resp = ingest_api(
                            ingest_source,
                            ingest_type,
                            ingest_chunking,
                            ingest_chunk_size,
                            ingest_chunk_overlap,
                            ingest_incremental,
                        )
                        st.success(resp.get("message", "Ingestion in progress…"))
                    except Exception as e:
                        st.error(f"Ingest failed: {e}")

    st.divider()

    # Clear history
    if st.button("🗑️ Clear chat & history", use_container_width=True):
        st.session_state.messages = []
        try:
            clear_history_api()
            st.success("History cleared.")
        except Exception:
            st.warning("Could not reach backend to clear server-side history.")
        st.rerun()


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("🤖 RAG Assistant")
st.caption("Ask questions about your indexed documents.")

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources if stored
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 Sources & retrieved chunks"):
                for i, (src, chunk) in enumerate(
                    zip(msg["sources"], msg.get("chunks", []))
                ):
                    st.markdown(f"**Source {i+1}:** `{src}`")
                    st.text(chunk[:500] + ("…" if len(chunk) > 500 else ""))
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask a question…"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API and display assistant response
    with st.chat_message("assistant"):
        if not health["pipeline_ready"]:
            answer = "⚠️ The vector store is not ready. Please ingest documents first."
            st.warning(answer)
            sources, chunks = [], []
        else:
            with st.spinner("Thinking…"):
                try:
                    result = query_api(
                        question=prompt,
                        k=k,
                        strategy=strategy,
                        use_history=st.session_state.use_history,
                    )
                    answer = result["answer"]
                    sources = result.get("sources", [])
                    chunks = result.get("chunks", [])
                except Exception as e:
                    answer = f"❌ Error: {e}"
                    sources, chunks = [], []

            st.markdown(answer)

            if sources:
                with st.expander("📎 Sources & retrieved chunks"):
                    for i, (src, chunk) in enumerate(zip(sources, chunks)):
                        st.markdown(f"**Source {i+1}:** `{src}`")
                        st.text(chunk[:500] + ("…" if len(chunk) > 500 else ""))
                        st.divider()

    # Persist to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "chunks": chunks,
    })
