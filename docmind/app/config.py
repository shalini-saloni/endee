"""
app/config.py
Centralised configuration — reads .env via python-dotenv.
"""
import os
from dotenv import load_dotenv

load_dotenv()

ENDEE_BASE_URL: str  = os.getenv("ENDEE_BASE_URL", "http://localhost:8080/api/v1")
ENDEE_AUTH_TOKEN: str = os.getenv("ENDEE_AUTH_TOKEN", "")

INDEX_NAME: str = os.getenv("INDEX_NAME", "docmind")
EMBED_DIM: int  = 384         

EMBED_MODEL: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

CHUNK_SIZE: int    = int(os.getenv("CHUNK_SIZE",    "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

TOP_K: int = int(os.getenv("TOP_K", "5"))

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY: str   = os.getenv("GROQ_API_KEY", "")

if GROQ_API_KEY:
    LLM_PROVIDER   = "groq"
    LLM_MODEL      = "llama-3.1-8b-instant"
    LLM_BASE_URL   = "https://api.groq.com/openai/v1"
    LLM_API_KEY    = GROQ_API_KEY
elif OPENAI_API_KEY:
    LLM_PROVIDER   = "openai"
    LLM_MODEL      = "gpt-4o-mini"
    LLM_BASE_URL   = "https://api.openai.com/v1"
    LLM_API_KEY    = OPENAI_API_KEY
else:
    LLM_PROVIDER   = None
    LLM_MODEL      = None
    LLM_BASE_URL   = None
    LLM_API_KEY    = None
