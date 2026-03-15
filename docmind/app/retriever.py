"""
app/retriever.py
Query → embedding → Endee ANN search → ranked chunks.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional

from app.embedder import embed_query
from app.endee_client import search
from app.config import TOP_K


def retrieve(
    query: str,
    top_k: int = TOP_K,
    doc_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k most relevant chunks for *query*.

    Args:
        query:    Natural-language question.
        top_k:    Number of results to return.
        doc_name: If given, restrict search to this document.

    Returns:
        List of dicts, each with:
            id         – Endee chunk ID
            similarity – cosine similarity score (0 – 1)
            text       – the chunk's raw text
            filename   – source file name
            page_number
    """
    query_vec = embed_query(query)
    raw_results = search(query_vec, top_k=top_k, doc_name=doc_name)

    # Flatten meta into the result for convenience
    results = []
    for item in raw_results:
        meta = item.get("meta") or {}
        results.append({
            "id":          item.get("id"),
            "similarity":  round(item.get("similarity", 0.0), 4),
            "text":        meta.get("text", ""),
            "filename":    meta.get("filename", ""),
            "page_number": meta.get("page_number", 0),
        })
    return results
