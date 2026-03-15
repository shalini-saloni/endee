"""
app/endee_client.py
Thin wrapper around the official Endee Python SDK.

Provides:
  - get_or_create_index()   – idempotent index bootstrap
  - upsert_chunks()         – bulk-insert pre-embedded chunks
  - search()                – ANN query with optional doc_name filter
"""
from __future__ import annotations
from functools import lru_cache
from typing import List, Dict, Any, Optional

from endee import Endee, Precision
from app.config import (
    ENDEE_BASE_URL,
    ENDEE_AUTH_TOKEN,
    INDEX_NAME,
    EMBED_DIM,
    TOP_K,
)


@lru_cache(maxsize=1)
def _get_client() -> Endee:
    """Return a cached Endee client pointed at the local server."""
    client = Endee(ENDEE_AUTH_TOKEN) if ENDEE_AUTH_TOKEN else Endee()
    client.set_base_url(ENDEE_BASE_URL)
    return client


def get_or_create_index():
    """
    Return (and optionally create) the docmind index.
    Using INT8 quantization for a good speed/accuracy trade-off.
    """
    client = _get_client()
    existing = {idx["name"] for idx in client.list_indexes()}
    if INDEX_NAME not in existing:
        print(f"[endee] Creating index '{INDEX_NAME}' (dim={EMBED_DIM}) …")
        client.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            space_type="cosine",
            precision=Precision.INT8,
        )
    return client.get_index(name=INDEX_NAME)


def upsert_chunks(chunks: List[Dict[str, Any]]) -> None:
    """
    Insert / update a list of chunk dicts into Endee.

    Each dict must have:
        id      : str          – unique chunk ID
        vector  : List[float]  – 384-dim embedding
        meta    : dict         – arbitrary metadata (text, source, page …)
        filter  : dict         – filterable fields (doc_name, page_number …)
    """
    index = get_or_create_index()
    index.upsert(chunks)
    print(f"[endee] Upserted {len(chunks)} chunk(s) into '{INDEX_NAME}'.")


def search(
    query_vector: List[float],
    top_k: int = TOP_K,
    doc_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Perform ANN search.

    If doc_name is given, restrict results to chunks from that document
    using Endee's built-in $eq filter.

    Returns a list of result dicts with keys:
        id, similarity, meta, filter
    """
    index = get_or_create_index()

    filter_clause: Optional[List[Dict]] = None
    if doc_name:
        filter_clause = [{"doc_name": {"$eq": doc_name}}]

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        ef=128,                          
        filter=filter_clause,
        filter_boost_percentage=20,       
    )
    return results


def describe_index() -> Dict[str, Any]:
    """Return stats about the current index."""
    index = get_or_create_index()
    return index.describe()
