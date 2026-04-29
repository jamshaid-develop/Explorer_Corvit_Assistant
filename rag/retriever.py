"""
rag/retriever.py — Semantic retrieval from ChromaDB
Uses ChromaDB default embeddings (no torch needed)
"""

from typing import List
from loguru import logger

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from rag.ingest import get_chroma_collection


def retrieve(query: str, top_k: int = None) -> List[dict]:
    if top_k is None:
        top_k = config.TOP_K_RESULTS

    coll = get_chroma_collection()

    if coll.count() == 0:
        logger.warning("[Retriever] Vector store is empty.")
        return []

    results = coll.query(
        query_texts=[query],
        n_results=min(top_k, coll.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":       doc,
            "source":     meta.get("source", "unknown"),
            "similarity": round(1 - dist, 3),
        })

    logger.info(f"[Retriever] {len(chunks)} chunks for: '{query[:60]}'")
    return chunks


def format_context(chunks: List[dict]) -> str:
    if not chunks:
        return "No specific Corvit data found for this query."
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source {i} — {chunk['source']} | relevance: {chunk['similarity']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)