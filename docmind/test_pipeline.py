"""
test_pipeline.py
────────────────
Quick smoke-test for the full DocMind pipeline.
Run this BEFORE the Streamlit UI to verify everything is wired correctly.

Usage:
    python test_pipeline.py
"""
import sys
import os

def ok(msg):   print(f"{msg}")
def fail(msg): print(f"{msg}"); sys.exit(1)
def info(msg): print(f"{msg}")
def section(title): print(f"\n{'─'*55}\n  {title}\n{'─'*55}")

section("1 · Environment")

if not os.path.exists(".env"):
    fail(".env not found. Run:  cp .env.example .env  then add your API key.")
ok(".env found")

from dotenv import load_dotenv
load_dotenv()

from app.config import (
    ENDEE_BASE_URL, INDEX_NAME, EMBED_MODEL,
    LLM_PROVIDER, LLM_MODEL, LLM_API_KEY
)
info(f"Endee URL  : {ENDEE_BASE_URL}")
info(f"Index name : {INDEX_NAME}")
info(f"Embed model: {EMBED_MODEL}")
info(f"LLM        : {LLM_PROVIDER} / {LLM_MODEL}")
if not LLM_API_KEY:
    fail("No LLM API key found. Set OPENAI_API_KEY or GROQ_API_KEY in .env")
ok("LLM API key found")

section("2 · Endee connection")

import requests as _req
try:
    r = _req.get("http://localhost:8080", timeout=5)
    ok(f"Endee reachable (HTTP {r.status_code})")
except Exception as e:
    fail(f"Cannot reach Endee at http://localhost:8080 — is Docker running?\n     {e}")

from app.endee_client import get_or_create_index, describe_index
index = get_or_create_index()
ok(f"Index '{INDEX_NAME}' ready")

section("3 · Embedding model")

from app.embedder import embed_texts, embed_query
sample = ["Hello, this is a test sentence.", "Vector databases are powerful."]
vecs = embed_texts(sample)
assert len(vecs) == 2, "Expected 2 vectors"
assert len(vecs[0]) == 384, f"Expected dim=384, got {len(vecs[0])}"
ok(f"Embedded {len(sample)} sentences → dim={len(vecs[0])}")

section("4 · Ingestion pipeline")

doc_path = "sample_docs/ai_overview.txt"
if not os.path.exists(doc_path):
    fail(f"Sample document not found: {doc_path}")

with open(doc_path, "rb") as f:
    content = f.read()

from app.ingest import ingest_document
n = ingest_document("ai_overview.txt", content)
ok(f"Ingested 'ai_overview.txt' → {n} chunks stored in Endee")

section("5 · Semantic retrieval")

from app.retriever import retrieve
query = "What is Retrieval-Augmented Generation?"
results = retrieve(query, top_k=3)
assert len(results) > 0, "Expected at least 1 result"
ok(f"Query returned {len(results)} chunk(s)")
for i, r in enumerate(results, 1):
    info(f"  [{i}] similarity={r['similarity']:.3f}  page={r['page_number']}")
    info(f"       {r['text'][:100]}…")

section("6 · RAG answer generation")

from app.rag import answer
result = answer(query="What is RAG and how does it work?", top_k=3)
ok(f"LLM answer generated ({len(result['answer'])} chars)")
ok(f"Model used: {result['model']}")
info(f"\n  Answer preview:\n  {result['answer'][:300]}…\n")
info(f"  Sources: {len(result['sources'])} chunk(s) cited")

section("7 · Index stats")

stats = describe_index()
ok(f"Index info: {stats}")

print(f"\n{'═'*55}")
print("All tests passed! DocMind pipeline is working.")
print(f"Run the UI:  streamlit run streamlit_app.py")
print(f"Run the API: uvicorn app.main:app --reload --port 8000")
print(f"{'═'*55}\n")
