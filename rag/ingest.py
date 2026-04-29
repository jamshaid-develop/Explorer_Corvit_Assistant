"""
rag/ingest.py — Document ingestion pipeline
Reads PDF / DOCX / TXT files, chunks them, embeds, and stores in ChromaDB
"""

import os
import hashlib
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from loguru import logger

# LangChain text splitter (light dependency — no LLM needed here)
from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_pdf(path: str) -> str:
    """Extract text from PDF using pypdf."""
    from pypdf import PdfReader
    reader = PdfReader(path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return text.strip()


def load_docx(path: str) -> str:
    """Extract text from DOCX."""
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def load_txt(path: str) -> str:
    """Read plain text file."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_file(path: str) -> str:
    """Auto-detect file type and load content."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext == ".docx":
        return load_docx(path)
    elif ext in (".txt", ".md"):
        return load_txt(path)
    else:
        logger.warning(f"[Ingest] Unsupported file type: {ext} — skipping {path}")
        return ""


# ─── Chunker ──────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> List[str]:
    """Split text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_text(text)


# ─── ChromaDB Client ──────────────────────────────────────────────────────────

_chroma_client: chromadb.Client = None
_collection: chromadb.Collection = None
_embedder: SentenceTransformer = None


def get_chroma_collection() -> chromadb.Collection:
    """Initialize and return the ChromaDB collection (singleton)."""
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(
            path=str(config.VECTORSTORE_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _chroma_client.get_or_create_collection(
            name="corvit_knowledge",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"[Ingest] ChromaDB collection loaded: {_collection.count()} docs")
    return _collection


def get_embedder() -> SentenceTransformer:
    """Load embedding model (singleton)."""
    global _embedder
    if _embedder is None:
        logger.info(f"[Ingest] Loading embedder: {config.EMBEDDING_MODEL}")
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder


# ─── Ingest Pipeline ──────────────────────────────────────────────────────────

def ingest_file(file_path: str) -> int:
    """
    Full pipeline: load → chunk → embed → store in ChromaDB.

    Args:
        file_path: Path to document file

    Returns:
        Number of chunks added
    """
    logger.info(f"[Ingest] Processing file: {file_path}")
    text = load_file(file_path)
    if not text:
        logger.warning(f"[Ingest] No text extracted from {file_path}")
        return 0

    chunks   = chunk_text(text)
    embedder = get_embedder()
    coll     = get_chroma_collection()

    # Generate unique IDs based on content hash to avoid duplicates
    ids        = []
    embeddings = []
    documents  = []
    metadatas  = []

    source_name = Path(file_path).name

    for i, chunk in enumerate(chunks):
        chunk_id = hashlib.md5(f"{source_name}_{i}_{chunk[:50]}".encode()).hexdigest()
        ids.append(chunk_id)
        embeddings.append(embedder.encode(chunk).tolist())
        documents.append(chunk)
        metadatas.append({"source": source_name, "chunk_index": i})

    # Upsert to avoid duplicates on re-ingest
    coll.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    logger.success(f"[Ingest] ✅ Added {len(chunks)} chunks from '{source_name}'")
    return len(chunks)


def ingest_directory(dir_path: str) -> int:
    """Ingest all supported files in a directory."""
    total = 0
    for file in Path(dir_path).glob("*"):
        if file.suffix.lower() in (".pdf", ".docx", ".txt", ".md"):
            total += ingest_file(str(file))
    return total


def ingest_text(text: str, source_name: str = "manual_input") -> int:
    """Ingest raw text string directly (e.g. website content)."""
    chunks   = chunk_text(text)
    embedder = get_embedder()
    coll     = get_chroma_collection()

    ids, embeddings, documents, metadatas = [], [], [], []
    for i, chunk in enumerate(chunks):
        chunk_id = hashlib.md5(f"{source_name}_{i}_{chunk[:50]}".encode()).hexdigest()
        ids.append(chunk_id)
        embeddings.append(embedder.encode(chunk).tolist())
        documents.append(chunk)
        metadatas.append({"source": source_name, "chunk_index": i})

    coll.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    logger.success(f"[Ingest] ✅ Added {len(chunks)} chunks from text source '{source_name}'")
    return len(chunks)


def get_collection_stats() -> dict:
    """Return stats about the current knowledge base."""
    coll = get_chroma_collection()
    count = coll.count()
    return {"total_chunks": count, "collection": "corvit_knowledge"}