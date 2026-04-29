"""
agent.py — Core orchestrator for Explorer Corvit Assistant
Ties together: Memory → RAG → Primary LLM → Fallback LLM → Response
"""

import json
import time
from loguru import logger

import config
from llm.primary_model  import PrimaryLLM
from llm.fallback_model import FallbackLLM
from rag.retriever      import retrieve, format_context
from memory.chat_memory import (
    create_session, save_message, get_recent_messages,
    get_session_summary, get_session_message_count,
)


class CorvitAgent:
    """
    The main AI agent for Explorer Corvit Assistant.

    Flow per query:
    1. Load short-term memory (recent messages) + long-term summary
    2. Retrieve relevant Corvit docs from vector store
    3. Build prompt: system + summary + history + context + user query
    4. Send to Primary LLM
    5. If Primary fails/weak → Fallback LLM
    6. Save messages to memory
    7. Return response + metadata
    """

    CONFIDENCE_THRESHOLD = 0.4   # Below this → trigger fallback
    SHORT_TERM_WINDOW    = 8     # Recent messages to include

    def __init__(self):
        self.primary  = PrimaryLLM()
        self.fallback = FallbackLLM()
        logger.info("[Agent] CorvitAgent initialized.")

    def chat(self, session_id: str, user_query: str) -> dict:
        """
        Main entry point. Returns response dict.

        Args:
            session_id:  Unique session identifier
            user_query:  User's input text

        Returns:
            dict with keys: answer, model_used, latency, success, context_used
        """
        start = time.time()

        # ── 1. Ensure session exists ──────────────────────────────────────────
        create_session(session_id)

        # ── 2. Load memory ────────────────────────────────────────────────────
        history = get_recent_messages(session_id, limit=self.SHORT_TERM_WINDOW)
        summary = get_session_summary(session_id)

        # ── 3. Retrieve relevant Corvit context ───────────────────────────────
        chunks        = retrieve(user_query)
        context_text  = format_context(chunks)
        context_used  = len(chunks) > 0

        # ── 4. Build LLM messages ─────────────────────────────────────────────
        messages = self._build_messages(
            user_query=user_query,
            history=history,
            context=context_text,
            summary=summary,
        )

        # ── 5. Primary LLM ────────────────────────────────────────────────────
        result = self.primary.generate(messages)

        # ── 6. Fallback if needed ──────────────────────────────────────────────
        fallback_used = False
        if not result["success"] or result["confidence"] < self.CONFIDENCE_THRESHOLD:
            logger.warning(
                f"[Agent] Primary failed/weak (conf={result['confidence']:.2f}). "
                f"Routing to fallback..."
            )
            result       = self.fallback.generate(messages)
            fallback_used = True

        # ── 7. Graceful degradation ───────────────────────────────────────────
        if not result["success"] or not result["text"]:
            answer = (
                "I'm sorry, I'm having trouble connecting right now. "
                "Please try again in a moment, or contact Corvit Systems directly at "
                "their official website."
            )
            result["text"] = answer

        answer = result["text"]
        total_latency = round(time.time() - start, 2)

        # ── 8. Save to memory ─────────────────────────────────────────────────
        save_message(session_id, "user",      user_query, latency=0.0)
        save_message(session_id, "assistant", answer,
                     model_used=result.get("model", ""),
                     latency=result.get("latency", 0.0))

        logger.success(
            f"[Agent] Done | session={session_id} | "
            f"model={result.get('model')} | latency={total_latency}s | "
            f"fallback={fallback_used}"
        )

        return {
            "answer":        answer,
            "model_used":    result.get("model", "unknown"),
            "latency":       total_latency,
            "success":       result.get("success", False),
            "context_used":  context_used,
            "fallback_used": fallback_used,
            "confidence":    result.get("confidence", 0.0),
        }

    def _build_messages(
        self,
        user_query: str,
        history: list,
        context: str,
        summary: str,
    ) -> list:
        """
        Assemble the full message list for the LLM:
        [system_message, ...history, user_message_with_context]
        """
        # System message
        system_content = config.SYSTEM_PROMPT
        if summary:
            system_content += f"\n\n## Conversation Summary (long-term memory):\n{summary}"

        messages = [{"role": "system", "content": system_content}]

        # Add recent chat history (short-term memory)
        messages.extend(history)

        # Build context-aware user message
        user_content = f"""{user_query}

---
## Relevant Corvit Knowledge Base:
{context}
"""
        messages.append({"role": "user", "content": user_content})
        return messages