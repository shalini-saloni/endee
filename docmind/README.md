# DocMind — RAG-powered Document Q&A using Endee Vector DB

> Upload any PDF or TXT document and ask natural language questions — answered using semantic retrieval from **Endee** and an LLM.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Endee](https://img.shields.io/badge/VectorDB-Endee-purple)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Problem Statement

Documents are knowledge silos. PDFs and text files contain dense, valuable information — research papers, legal contracts, product manuals — but searching them traditionally means exact keyword matching or manual reading.

**DocMind solves this** by converting documents into semantic vector embeddings stored in Endee, enabling meaning-based retrieval. When you ask a question, the most relevant passages are retrieved by cosine similarity and fed to an LLM that generates a grounded, citation-backed answer.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                       │
│                                                                 │
│  PDF / TXT ──► Text Extraction ──► Recursive Chunking          │
│                (PyMuPDF)           (LangChain splitter)         │
│                      │                                          │
│                      ▼                                          │
│             Sentence Transformer                                │
│             (all-MiniLM-L6-v2, 384-dim)                        │
│                      │                                          │
│                      ▼                                          │
│          ┌───────────────────────┐                              │
│          │   Endee Vector DB     │  ◄── upsert(id, vector,     │
│          │   (cosine, INT8)      │           meta, filter)      │
│          └───────────────────────┘                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        QUERY PIPELINE                           │
│                                                                 │
│  User Question ──► Embed Query ──► Endee ANN Search            │
│                    (384-dim)       (top-k, $eq filter)          │
│                                         │                       │
│                                         ▼                       │
│                              Top-K Relevant Chunks              │
│                                         │                       │
│                                         ▼                       │
│                         Augmented Prompt Assembly               │
│                                         │                       │
│                                         ▼                       │
│                    LLM (OpenAI GPT-4o-mini / Groq Llama3)      │
│                                         │                       │
│                                         ▼                       │
│                      Answer + Source Citations                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## How Endee is Used

Endee is the **core retrieval engine** of DocMind. It replaces traditional keyword search with Approximate Nearest Neighbor (ANN) vector search.

| Operation | Where in code | What it does |
|---|---|---|
| `create_index()` | `app/endee_client.py` | Creates a 384-dim cosine index with INT8 quantization |
| `index.upsert()` | `app/endee_client.py` | Stores chunk embeddings with `meta` (text, page) and `filter` (doc_name) |
| `index.query()` | `app/endee_client.py` | ANN search with optional `$eq` filter to scope by document |
| `index.describe()` | `app/endee_client.py` | Returns live index stats (vector count, dimensions) |

**Key Endee features leveraged:**
- **Cosine similarity** — ideal for normalised sentence-transformer embeddings
- **INT8 quantization** — 4× memory reduction with minimal accuracy loss
- **Payload filtering** — `{"doc_name": {"$eq": "report"}}` restricts search to one document
- **`filter_boost_percentage`** — compensates for filtered-out HNSW candidates

---

## Project Structure

```
docmind/
├── app/
│   ├── __init__.py
│   ├── config.py          # All settings loaded from .env
│   ├── embedder.py        # Singleton sentence-transformer wrapper
│   ├── endee_client.py    # Endee SDK wrapper (create, upsert, search)
│   ├── ingest.py          # PDF/TXT → chunks → embeddings → Endee
│   ├── retriever.py       # Query → embed → Endee search → chunks
│   ├── rag.py             # Context assembly → LLM → answer
│   └── main.py            # FastAPI app (/upload, /query, /index/info)
├── streamlit_app.py       # Streamlit chat UI
├── docker-compose.yml     # Endee server
├── requirements.txt
├── .env.example
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Vector Database | **Endee** (local Docker, Python SDK) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (384-dim) |
| LLM | OpenAI `gpt-4o-mini` or Groq `llama-3.1-8b-instant` (free) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| PDF Parsing | PyMuPDF (`fitz`) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Config | python-dotenv |

---

## Setup & Running

### Prerequisites

- Python 3.8+
- Docker + Docker Compose v2
- An OpenAI **or** Groq API key (Groq is free: https://console.groq.com)

---

### Step 1 — Fork & Star Endee (Mandatory)

```bash
# 1. Go to https://github.com/endee-io/endee
# 2. Click ⭐ Star
# 3. Click Fork → your GitHub account

# Then clone YOUR fork:
git clone https://github.com/<YOUR_USERNAME>/endee.git
cd endee
```

---

### Step 2 — Go To The Project

```bash
cd docmind
```

---

### Step 3 — Start the Endee Vector DB

```bash
# From the docmind/ directory (docker-compose.yml is here)
docker compose up -d

# Verify it's running
docker ps
# → you should see: endee-server   Up   0.0.0.0:8080->8080/tcp

# Open the Endee dashboard (optional)
open http://localhost:8080
```

---

### Step 4 — Python Environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

### Step 5 — Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Endee
ENDEE_BASE_URL=http://localhost:8080/api/v1
ENDEE_AUTH_TOKEN=

# OpenAI
OPENAI_API_KEY=

# Groq
GROQ_API_KEY=gsk_xxxxx

# LLM
LLM_MODEL=llama-3.1-8b-instant

# App
INDEX_NAME=docmind
EMBED_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=5
```

---

### Step 6 — Start the FastAPI Backend

```bash
uvicorn app.main:app --reload --port 8000
```

API docs available at: **http://localhost:8000/docs**

---

### Step 7 — Start the Streamlit Frontend

Open a second terminal:

```bash
source venv/bin/activate
streamlit run streamlit_app.py
```

Open: **http://localhost:8501**

---

## Usage

### Via Streamlit UI

1. Open **http://localhost:8501**
2. Use the sidebar to **upload a PDF or TXT file** → click **Ingest Document**
3. Wait for the success message (e.g., *"243 chunks stored"*)
4. Type your question in the chat input
5. DocMind returns an answer with **source citations** (filename + page number + similarity score)

### Via API (curl / Postman)

**Upload a document:**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/your/document.pdf"
```

**Ask a question (all documents):**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main conclusion of the paper?", "top_k": 5}'
```

**Ask a question (scoped to one document):**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key risks?", "doc_name": "annual_report", "top_k": 5}'
```

**Check index stats:**
```bash
curl http://localhost:8000/index/info
```

---

## Example Q&A

| Question | Answer Summary |
|---|---|
| *"What is Retrieval-Augmented Generation?"* | Retrieves relevant chunks from **ai_overview.txt** and explains how RAG combines retrieval with LLM generation. |
| *"What distance metrics do vector databases use?"* | Retrieves content from **vector_databases_guide.txt** describing cosine similarity, Euclidean distance (L2), and dot product. |
| *"What is Python mainly used for?"* | Retrieves information from **python_guide.txt** explaining Python’s readability and common applications. |
| *"Explain vector embeddings."* | Retrieves embedding explanation chunks and summarizes how data is represented in high-dimensional vector space. |

---

## Architecture Decisions & Trade-offs

| Decision | Rationale |
|---|---|
| `all-MiniLM-L6-v2` embeddings | Small (80MB), fast, 384-dim — great for on-device use |
| INT8 quantization in Endee | 4× lower memory, <1% accuracy drop on cosine retrieval |
| Chunk size 500 / overlap 50 | Balances context richness vs retrieval precision |
| `filter_boost_percentage=20` | Compensates for filtered-out HNSW candidates when scoping to one doc |
| OpenAI-compatible client for LLM | Single code path works for both OpenAI and Groq |
| SHA-1 chunk IDs | Deterministic — re-ingesting the same doc is safe (idempotent upsert) |

---

## Stopping & Cleanup

```bash
# Stop Endee (keeps data)
docker compose down

# Stop Endee AND wipe all stored vectors
docker compose down -v
```

---

