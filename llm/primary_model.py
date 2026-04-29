"""
llm/primary_model.py — Primary Groq LLM handler
Uses llama-3.3-70b-versatile (fast, high-quality)
"""

from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger
import time

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


class PrimaryLLM:
    """
    Handles calls to the primary Groq LLM.
    Includes retry logic, timeout handling, and confidence scoring.
    """

    def __init__(self):
        self.client     = Groq(api_key=config.GROQ_API_KEY)
        self.model      = config.PRIMARY_MODEL
        self.max_tokens = 1024
        self.temperature = 0.7

    @retry(
        stop=stop_after_attempt(config.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def generate(self, messages: list[dict]) -> dict:
        """
        Send messages to primary LLM and return response dict.

        Args:
            messages: List of {role, content} dicts (system + history + user)

        Returns:
            dict with keys: text, model, latency, success, confidence
        """
        start = time.time()
        try:
            logger.info(f"[PrimaryLLM] Calling {self.model} ...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=config.REQUEST_TIMEOUT,
            )
            text    = response.choices[0].message.content.strip()
            latency = round(time.time() - start, 2)

            if not text or len(text) < 10:
                logger.warning("[PrimaryLLM] Empty or too-short response received.")
                return self._failed_result(latency)

            confidence = self._estimate_confidence(text)
            logger.success(f"[PrimaryLLM] Response ({latency}s, conf={confidence:.2f}): {text[:80]}...")

            return {
                "text":       text,
                "model":      self.model,
                "latency":    latency,
                "success":    True,
                "confidence": confidence,
            }

        except Exception as e:
            latency = round(time.time() - start, 2)
            logger.error(f"[PrimaryLLM] Error: {e}")
            return self._failed_result(latency, error=str(e))

    def _estimate_confidence(self, text: str) -> float:
        """
        Heuristic confidence score based on response quality signals.
        Returns float between 0.0 and 1.0.
        """
        score = 1.0
        low_conf_phrases = [
            "i don't know", "i'm not sure", "i cannot", "no information",
            "not available", "pata nahi", "معلوم نہیں"
        ]
        for phrase in low_conf_phrases:
            if phrase.lower() in text.lower():
                score -= 0.3
        if len(text) < 50:
            score -= 0.2
        return max(0.0, min(1.0, score))

    def _failed_result(self, latency: float, error: str = "") -> dict:
        return {
            "text":       "",
            "model":      self.model,
            "latency":    latency,
            "success":    False,
            "confidence": 0.0,
            "error":      error,
        }