"""
llm/fallback_model.py — Fallback Groq LLM handler
Uses mixtral-8x7b-32768 as backup when primary fails or gives weak output
"""

from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger
import time

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


class FallbackLLM:
    """
    Backup LLM that activates when PrimaryLLM fails, times out,
    or returns a low-confidence / empty response.
    """

    def __init__(self):
        self.client      = Groq(api_key=config.GROQ_API_KEY)
        self.model       = config.FALLBACK_MODEL
        self.max_tokens  = 1024
        self.temperature = 0.6   # Slightly lower temp for more factual output

    @retry(
        stop=stop_after_attempt(2),                            # Fewer retries for fallback
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def generate(self, messages: list[dict]) -> dict:
        """
        Send messages to fallback LLM and return response dict.
        Same interface as PrimaryLLM.generate().
        """
        start = time.time()
        try:
            logger.info(f"[FallbackLLM] Calling {self.model} ...")
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
                logger.warning("[FallbackLLM] Empty response received.")
                return self._failed_result(latency)

            logger.success(f"[FallbackLLM] Response ({latency}s): {text[:80]}...")
            return {
                "text":       text,
                "model":      self.model,
                "latency":    latency,
                "success":    True,
                "confidence": 0.8,   # fallback answers are usually decent
            }

        except Exception as e:
            latency = round(time.time() - start, 2)
            logger.error(f"[FallbackLLM] Error: {e}")
            return self._failed_result(latency, error=str(e))

    def _failed_result(self, latency: float, error: str = "") -> dict:
        return {
            "text":       "",
            "model":      self.model,
            "latency":    latency,
            "success":    False,
            "confidence": 0.0,
            "error":      error,
        }