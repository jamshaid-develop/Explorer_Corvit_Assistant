"""
rag/ingest.py — Document ingestion pipeline
Uses ChromaDB default embeddings (no sentence-transformers / torch needed)
"""

import hashlib
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_pdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def load_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_file(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":   return load_pdf(path)
    elif ext == ".docx": return load_docx(path)
    elif ext in (".txt", ".md"): return load_txt(path)
    else:
        logger.warning(f"[Ingest] Unsupported file type: {ext}")
        return ""


# ─── Chunker ──────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_text(text)


# ─── ChromaDB — uses built-in default embeddings (no torch) ──────────────────

_collection = None

def get_chroma_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(
            path=str(config.VECTORSTORE_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        # Use ChromaDB's default embedding function (no extra dependencies)
        ef = embedding_functions.DefaultEmbeddingFunction()
        _collection = client.get_or_create_collection(
            name="corvit_knowledge",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"[Ingest] Collection loaded: {_collection.count()} docs")
    return _collection


# ─── Ingest Pipeline ──────────────────────────────────────────────────────────

def ingest_file(file_path: str) -> int:
    logger.info(f"[Ingest] Processing: {file_path}")
    text = load_file(file_path)
    if not text:
        return 0

    chunks = chunk_text(text)
    coll   = get_chroma_collection()
    source = Path(file_path).name

    ids, documents, metadatas = [], [], []
    for i, chunk in enumerate(chunks):
        chunk_id = hashlib.md5(f"{source}_{i}_{chunk[:50]}".encode()).hexdigest()
        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append({"source": source, "chunk_index": i})

    coll.upsert(ids=ids, documents=documents, metadatas=metadatas)
    logger.success(f"[Ingest] ✅ {len(chunks)} chunks from '{source}'")
    return len(chunks)


def ingest_text(text: str, source_name: str = "manual_input") -> int:
    chunks = chunk_text(text)
    coll   = get_chroma_collection()
    ids, documents, metadatas = [], [], []
    for i, chunk in enumerate(chunks):
        chunk_id = hashlib.md5(f"{source_name}_{i}_{chunk[:50]}".encode()).hexdigest()
        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append({"source": source_name, "chunk_index": i})
    coll.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(chunks)


def ingest_directory(dir_path: str) -> int:
    total = 0
    for file in Path(dir_path).glob("*"):
        if file.suffix.lower() in (".pdf", ".docx", ".txt", ".md"):
            total += ingest_file(str(file))
    return total


def get_collection_stats() -> dict:
    coll = get_chroma_collection()
    return {"total_chunks": coll.count(), "collection": "corvit_knowledge"}