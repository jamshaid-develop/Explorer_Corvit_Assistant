"""
memory/chat_memory.py — Persistent chat memory using SQLite
Stores full conversation history per session with short-term + long-term memory
"""

import sqlite3
import json
import time
from datetime import datetime
from typing import List, Optional
from pathlib import Path
from loguru import logger

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


# ─── Database Setup ───────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    """Return a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL,
                summary      TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS messages (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT NOT NULL,
                role         TEXT NOT NULL,       -- 'user' | 'assistant'
                content      TEXT NOT NULL,
                model_used   TEXT DEFAULT '',
                latency      REAL DEFAULT 0.0,
                timestamp    TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, id);
        """)
    logger.info("[Memory] Database initialized.")


# ─── Session Management ───────────────────────────────────────────────────────

def create_session(session_id: str) -> str:
    """Create a new session. Returns session_id."""
    now = datetime.utcnow().isoformat()
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, created_at, updated_at) VALUES (?, ?, ?)",
            (session_id, now, now),
        )
    logger.info(f"[Memory] Session created/resumed: {session_id}")
    return session_id


def list_sessions() -> List[dict]:
    """Return all sessions ordered by most recent activity."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT session_id, created_at, updated_at, summary FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: str):
    """Delete a session and all its messages."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
    logger.info(f"[Memory] Session deleted: {session_id}")


# ─── Message Storage ──────────────────────────────────────────────────────────

def save_message(
    session_id: str,
    role: str,
    content: str,
    model_used: str = "",
    latency: float = 0.0,
):
    """Save a single message to the database."""
    now = datetime.utcnow().isoformat()
    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO messages (session_id, role, content, model_used, latency, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, role, content, model_used, latency, now),
        )
        conn.execute(
            "UPDATE sessions SET updated_at=? WHERE session_id=?",
            (now, session_id),
        )


def get_recent_messages(session_id: str, limit: int = 10) -> List[dict]:
    """
    Retrieve the most recent N messages for a session (short-term memory).
    Returns list of {role, content} dicts — ready for LLM message format.
    """
    with _get_conn() as conn:
        rows = conn.execute(
            """SELECT role, content FROM messages
               WHERE session_id=?
               ORDER BY id DESC LIMIT ?""",
            (session_id, limit),
        ).fetchall()
    # Reverse to chronological order
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def get_all_messages(session_id: str) -> List[dict]:
    """Return full conversation history for a session."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content, model_used, latency, timestamp FROM messages WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_session_message_count(session_id: str) -> int:
    """Return number of messages in a session."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE session_id=?", (session_id,)
        ).fetchone()
    return row["cnt"] if row else 0


def save_session_summary(session_id: str, summary: str):
    """Save an AI-generated summary for long-term memory."""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET summary=? WHERE session_id=?",
            (summary, session_id),
        )


def get_session_summary(session_id: str) -> str:
    """Retrieve the long-term summary for a session."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT summary FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
    return row["summary"] if row else ""


# ─── Initialize on import ────────────────────────────────────────────────────
init_db()