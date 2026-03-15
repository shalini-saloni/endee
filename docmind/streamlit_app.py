"""
streamlit_app.py
DocMind — Streamlit frontend.

Run with:
    streamlit run streamlit_app.py
"""
from pathlib import Path
import streamlit as st
import requests
import os

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

BASE_DIR = Path(__file__).resolve().parent
logo_path = BASE_DIR / "assets" / "logo-light.svg"

st.set_page_config(
    page_title="DocMind – Document Q&A",
    page_icon="🧠",
    layout="wide",
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []    # list of {"role", "content", "sources"}
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []   # list of doc_name strings

with st.sidebar:
    st.image(str(logo_path), width=180)

    st.title("DocMind")
    st.caption("RAG-powered Document Q&A")

    st.divider()
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader(
        "PDF or TXT", type=["pdf", "txt", "md"], label_visibility="collapsed"
    )
    if uploaded_file:
        if st.button("Ingest Document", use_container_width=True):
            with st.spinner("Extracting, chunking & indexing …"):
                try:
                    resp = requests.post(
                        f"{API_BASE}/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                        timeout=120,
                    )
                    if resp.ok:
                        data = resp.json()
                        st.success(
                            f"Ingested **{uploaded_file.name}**  \n"
                            f"{data['chunks']} chunks stored."
                        )
                        doc_name = data.get("doc_name", uploaded_file.name)
                        if doc_name not in st.session_state.uploaded_docs:
                            st.session_state.uploaded_docs.append(doc_name)
                    else:
                        st.error(f"Error: {resp.json().get('detail', resp.text)}")
                except requests.exceptions.ConnectionError:
                    st.error(
                        "Cannot reach the API. Is `uvicorn app.main:app` running?"
                    )

    st.divider()
    st.subheader("Filter by document")
    doc_options = ["All documents"] + st.session_state.uploaded_docs
    selected_doc = st.selectbox("Scope search to:", doc_options)

    st.divider()
    st.subheader("Settings")
    top_k = st.slider("Retrieved chunks (top-k)", 1, 10, 5)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    # Index stats
    st.divider()
    if st.button("Index info", use_container_width=True):
        try:
            info = requests.get(f"{API_BASE}/index/info", timeout=10).json()
            st.json(info)
        except Exception as e:
            st.error(str(e))

# ── main chat area ─────────────────────────────────────────────────────────────
st.title("DocMind — Ask your Documents")
st.caption(
    "Upload a PDF or TXT in the sidebar, then ask anything about it below. "
    "Powered by **Endee Vector DB** + **sentence-transformers** + LLM."
)

# Render chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**[{i}]** `{src['filename']}` — Page {src['page_number']} "
                        f"| Score: `{src['similarity']:.3f}`"
                    )
                    st.caption(src["snippet"])
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask a question about your documents …"):
    # Display user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Searching & generating answer …"):
            payload = {
                "question": prompt,
                "top_k": top_k,
                "doc_name": None if selected_doc == "All documents" else selected_doc,
            }
            try:
                resp = requests.post(
                    f"{API_BASE}/query", json=payload, timeout=60
                )
                if resp.ok:
                    data = resp.json()
                    answer_text = data.get("answer", "No answer returned.")
                    sources = data.get("sources", [])
                    model = data.get("model", "")
                else:
                    answer_text = f"API error: {resp.json().get('detail', resp.text)}"
                    sources = []
                    model = ""
            except requests.exceptions.ConnectionError:
                answer_text = "Cannot reach the API. Is `uvicorn app.main:app` running?"
                sources = []
                model = ""

        st.markdown(answer_text)
        if model:
            st.caption(f"Model: `{model}`")
        if sources:
            with st.expander(f"Sources ({len(sources)})"):
                for i, src in enumerate(sources, 1):
                    st.markdown(
                        f"**[{i}]** `{src['filename']}` — Page {src['page_number']} "
                        f"| Score: `{src['similarity']:.3f}`"
                    )
                    st.caption(src["snippet"])
                    st.divider()

    # Save to history
    st.session_state.chat_history.append({
        "role":    "assistant",
        "content": answer_text,
        "sources": sources,
    })
