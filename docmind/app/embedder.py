"""
app/embedder.py
Singleton wrapper around sentence-transformers.
Loads the model once and caches it for the lifetime of the process.
"""
from __future__ import annotations
from functools import lru_cache
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import EMBED_MODEL


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load (and cache) the embedding model on first call."""
    print(f"[embedder] Loading model '{EMBED_MODEL}' …")
    return SentenceTransformer(EMBED_MODEL)


def embed_texts(texts: Union[str, List[str]]) -> List[List[float]]:
    """
    Embed one or more strings.

    Returns a list of float lists – one per input string.
    Shape: (N, 384) for all-MiniLM-L6-v2.
    """
    if isinstance(texts, str):
        texts = [texts]
    model = _load_model()
    vectors: np.ndarray = model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(query: str) -> List[float]:
    """Convenience wrapper for a single query string."""
    return embed_texts(query)[0]
