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
    client = Endee(ENDEE_AUTH_TOKEN) if ENDEE_AUTH_TOKEN else Endee()
    client.set_base_url(ENDEE_BASE_URL)
    return client


def get_or_create_index():
    client = _get_client()

    try:
        indexes = client.list_indexes()

        existing = set()
        for idx in indexes:
            if isinstance(idx, dict):
                existing.add(idx.get("name"))
            else:
                existing.add(idx)

        if INDEX_NAME not in existing:
            print(f"[endee] Creating index '{INDEX_NAME}' (dim={EMBED_DIM}) …")
            client.create_index(
                name=INDEX_NAME,
                dimension=EMBED_DIM,
                space_type="cosine",
                precision=Precision.INT8,
            )
        else:
            print(f"[endee] Using existing index '{INDEX_NAME}'")

    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"[endee] Index '{INDEX_NAME}' already exists, using it")
        else:
            raise

    return client.get_index(name=INDEX_NAME)


def upsert_chunks(chunks: List[Dict[str, Any]]) -> None:
    index = get_or_create_index()
    index.upsert(chunks)
    print(f"[endee] Upserted {len(chunks)} chunk(s) into '{INDEX_NAME}'.")


def search(
    query_vector: List[float],
    top_k: int = TOP_K,
    doc_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    index = get_or_create_index()

    filter_clause = None
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
    index = get_or_create_index()
    return index.describe()