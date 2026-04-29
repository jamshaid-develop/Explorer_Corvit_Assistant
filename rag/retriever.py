"""
rag/retriever.py — Semantic retrieval from ChromaDB vector store
Finds the most relevant Corvit knowledge chunks for a given query
"""

from typing import List
from loguru import logger

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from rag.ingest import get_chroma_collection, get_embedder


def retrieve(query: str, top_k: int = None) -> List[dict]:
    """
    Retrieve the most semantically relevant chunks for a query.

    Args:
        query:  User's question
        top_k:  Number of chunks to retrieve (defaults to config value)

    Returns:
        List of dicts: [{text, source, score}, ...]
    """
    if top_k is None:
        top_k = config.TOP_K_RESULTS

    embedder = get_embedder()
    coll     = get_chroma_collection()

    if coll.count() == 0:
        logger.warning("[Retriever] Vector store is empty — no context available.")
        return []

    query_embedding = embedder.encode(query).tolist()

    results = coll.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, coll.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # Convert cosine distance to similarity score (0–1, higher = better)
        similarity = round(1 - dist, 3)
        chunks.append({
            "text":       doc,
            "source":     meta.get("source", "unknown"),
            "similarity": similarity,
        })

    logger.info(f"[Retriever] Retrieved {len(chunks)} chunks for query: '{query[:60]}...'")
    return chunks


def format_context(chunks: List[dict]) -> str:
    """
    Format retrieved chunks into a clean context string for the LLM prompt.
    """
    if not chunks:
        return "No specific Corvit data found for this query."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source {i} — {chunk['source']} | relevance: {chunk['similarity']}]\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)