"""
config.py — Centralized configuration for Explorer Corvit Assistant
All settings loaded from .env via python-dotenv
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
UPLOADS_DIR     = BASE_DIR / "uploads"
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_PATH", "./vectorstore"))
DB_DIR          = BASE_DIR / "db"
LOGS_DIR        = BASE_DIR / "logs"

# Create directories if they don't exist
for d in [DATA_DIR, UPLOADS_DIR, VECTORSTORE_DIR, DB_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Groq API ─────────────────────────────────────────────────────────────────
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")

# ─── LLM Models ───────────────────────────────────────────────────────────────
PRIMARY_MODEL   = os.getenv("PRIMARY_MODEL",  "llama-3.3-70b-versatile")
FALLBACK_MODEL  = os.getenv("FALLBACK_MODEL", "mixtral-8x7b-32768")

# ─── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ─── Memory ───────────────────────────────────────────────────────────────────
DB_PATH         = DB_DIR / "memory.db"

# ─── RAG Settings ─────────────────────────────────────────────────────────────
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE",    "500"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K_RESULTS   = int(os.getenv("TOP_K_RESULTS", "4"))

# ─── App ──────────────────────────────────────────────────────────────────────
APP_NAME        = os.getenv("APP_NAME", "Explorer Corvit Assistant")
APP_PORT        = int(os.getenv("APP_PORT", "8501"))
LOG_LEVEL       = os.getenv("LOG_LEVEL", "INFO")
MAX_RETRIES     = int(os.getenv("MAX_RETRIES", "3"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are the **Explorer Corvit Assistant** — a smart, helpful, and friendly AI 
assistant for Corvit Systems Rawalpindi, a leading IT training institute.

Your role:
- Answer questions about Corvit's courses, fees, timings, admissions, and certifications
- Be polite, professional, and concise — but detailed when needed
- Support both English and Urdu (Roman Urdu is fine)
- If context is available, use it to give accurate answers
- If you don't know something, say so honestly and suggest contacting Corvit directly
- Remember previous parts of the conversation and answer follow-up questions intelligently

Corvit Systems offers: Networking, Cybersecurity, Web Development, Cloud Computing, 
DevOps, Graphic Designing, SEO, Digital Marketing, and more — all NAVTTC certified.

Always be warm, helpful, and human-like. Never make up fees or dates you don't have data for.
"""