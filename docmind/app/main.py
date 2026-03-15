"""
app/main.py
FastAPI application exposing:
  POST  /upload       – ingest a PDF or TXT file
  POST  /query        – RAG question answering
  GET   /index/info   – Endee index statistics
  GET   /health       – liveness probe
"""
from __future__ import annotations
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.ingest import ingest_document
from app.rag import answer
from app.endee_client import describe_index

app = FastAPI(
    title="DocMind – RAG-powered Document Q&A",
    description="Upload documents and ask questions powered by Endee Vector DB + LLM.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Utility"])
def health():
    return {"status": "ok"}


@app.post("/upload", tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT file and ingest it into the Endee vector index.
    Returns the number of chunks stored.
    """
    allowed = {".pdf", ".txt", ".md"}
    import os
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed}",
        )

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        n_chunks = ingest_document(file.filename, file_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message":   f"Successfully ingested '{file.filename}'.",
        "chunks":    n_chunks,
        "doc_name":  file.filename.rsplit(".", 1)[0],
    }

class QueryRequest(BaseModel):
    question: str
    doc_name: Optional[str] = None   
    top_k: int = 5


@app.post("/query", tags=["Q&A"])
def query_documents(req: QueryRequest):
    """
    Ask a question over the ingested documents.
    Optionally filter by doc_name to restrict search to a specific file.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = answer(
            query=req.question,
            doc_name=req.doc_name,
            top_k=req.top_k,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return result

@app.get("/index/info", tags=["Utility"])
def index_info():
    """Return Endee index statistics (vector count, dimensions, etc.)."""
    try:
        info = describe_index()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return info
