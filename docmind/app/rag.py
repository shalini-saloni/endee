"""
app/rag.py
RAG pipeline:
  retrieved chunks  →  prompt assembly  →  LLM  →  answer + citations
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional

from openai import OpenAI

from app.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_PROVIDER
from app.retriever import retrieve


def _get_llm_client() -> OpenAI:
    if not LLM_API_KEY:
        raise RuntimeError(
            "No LLM API key found. Set OPENAI_API_KEY or GROQ_API_KEY in .env"
        )
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


_SYSTEM_PROMPT = """You are DocMind, an expert document assistant.
Answer the user's question using ONLY the context passages provided below.
If the answer is not in the context, say: "I couldn't find relevant information in the uploaded documents."
Be concise, accurate, and cite the source (filename + page) at the end of your answer."""


def _build_context(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] Source: {c['filename']} | Page {c['page_number']} "
            f"| Similarity: {c['similarity']:.3f}\n{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


def answer(
    query: str,
    doc_name: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Full RAG inference.

    Returns:
        {
            "answer":  str,
            "sources": [ {filename, page_number, similarity, snippet}, … ],
            "model":   str,
        }
    """

    chunks = retrieve(query, top_k=top_k, doc_name=doc_name)
    if not chunks:
        return {
            "answer":  "No relevant documents found. Please upload a document first.",
            "sources": [],
            "model":   LLM_MODEL or "none",
        }

    context = _build_context(chunks)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    client = _get_llm_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    answer_text = response.choices[0].message.content.strip()

    sources = [
        {
            "filename":    c["filename"],
            "page_number": c["page_number"],
            "similarity":  c["similarity"],
            "snippet":     c["text"][:200] + ("…" if len(c["text"]) > 200 else ""),
        }
        for c in chunks
    ]

    return {
        "answer":  answer_text,
        "sources": sources,
        "model":   LLM_MODEL,
    }
