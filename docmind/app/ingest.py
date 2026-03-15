"""
app/ingest.py
Document ingestion pipeline:
  PDF / TXT  →  text extraction
             →  recursive chunking
             →  sentence-transformer embeddings
             →  upsert into Endee
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
from typing import List, Tuple

import fitz  
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.embedder import embed_texts
from app.endee_client import upsert_chunks


def _extract_pdf(file_bytes: bytes) -> List[Tuple[int, str]]:
    """Return list of (page_number, page_text) tuples from PDF bytes."""
    pages: List[Tuple[int, str]] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append((page_num, text))
    return pages


def _extract_txt(file_bytes: bytes) -> List[Tuple[int, str]]:
    """Treat the whole text file as a single 'page'."""
    text = file_bytes.decode("utf-8", errors="replace").strip()
    return [(1, text)] if text else []


def extract_text(filename: str, file_bytes: bytes) -> List[Tuple[int, str]]:
    """Dispatch to the right extractor based on file extension."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(file_bytes)
    elif ext in (".txt", ".md"):
        return _extract_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {ext!r}. Use PDF or TXT.")


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def _chunk_page(page_text: str) -> List[str]:
    return _splitter.split_text(page_text)


def _make_chunk_id(doc_name: str, page: int, chunk_index: int, text: str) -> str:
    """Deterministic ID so re-ingesting the same document is idempotent."""
    raw = f"{doc_name}|p{page}|c{chunk_index}|{text[:64]}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def ingest_document(filename: str, file_bytes: bytes) -> int:
    """
    Full ingestion pipeline.
    Returns the total number of chunks upserted.
    """
    doc_name = Path(filename).stem  

    pages = extract_text(filename, file_bytes)
    if not pages:
        raise ValueError("No extractable text found in the document.")

    all_chunks_data: List[dict] = []
    for page_num, page_text in pages:
        chunks = _chunk_page(page_text)
        for i, chunk_text in enumerate(chunks):
            all_chunks_data.append({
                "text":       chunk_text,
                "doc_name":   doc_name,
                "filename":   filename,
                "page_number": page_num,
                "chunk_index": i,
            })

    if not all_chunks_data:
        raise ValueError("Document produced 0 chunks after splitting.")

    BATCH = 64
    vectors_flat: List[List[float]] = []
    texts_only = [d["text"] for d in all_chunks_data]
    for start in range(0, len(texts_only), BATCH):
        batch_texts = texts_only[start : start + BATCH]
        vectors_flat.extend(embed_texts(batch_texts))

    upsert_payload = []
    for data, vec in zip(all_chunks_data, vectors_flat):
        chunk_id = _make_chunk_id(
            data["doc_name"],
            data["page_number"],
            data["chunk_index"],
            data["text"],
        )
        upsert_payload.append({
            "id":     chunk_id,
            "vector": vec,
            "meta": {
                "text":        data["text"],
                "filename":    data["filename"],
                "page_number": data["page_number"],
                "chunk_index": data["chunk_index"],
            },
            "filter": {
                "doc_name": data["doc_name"],
            },
        })

    upsert_chunks(upsert_payload)
    return len(upsert_payload)
