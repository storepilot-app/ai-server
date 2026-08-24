import json
import logging
import re
import threading
import time
from time import perf_counter
import urllib.error
import urllib.request

import numpy as np

from app.category_matcher.config.settings import (
    SOLAR_EMBEDDING_API_BASE_URL,
    SOLAR_EMBEDDING_API_KEY,
    SOLAR_EMBEDDING_BATCH_SIZE,
    SOLAR_EMBEDDING_DIMENSIONS,
    SOLAR_EMBEDDING_PASSAGE_MODEL,
    SOLAR_EMBEDDING_QUERY_MODEL,
    SOLAR_EMBEDDING_REQUESTS_PER_MINUTE,
    SOLAR_EMBEDDING_TIMEOUT_SECONDS,
)


logger = logging.getLogger("uvicorn.error").getChild("storepilot.embedding_api")
_rate_limit_lock = threading.Lock()
_next_request_at = 0.0


class SolarEmbeddingProvider:
    provider_name = "solar"

    @property
    def model_name(self) -> str:
        return f"{SOLAR_EMBEDDING_QUERY_MODEL}+{SOLAR_EMBEDDING_PASSAGE_MODEL}"

    @property
    def cache_key(self) -> str:
        value = f"{self.provider_name}-{self.model_name}-{SOLAR_EMBEDDING_DIMENSIONS}"
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower()

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.embed_queries(texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, SOLAR_EMBEDDING_QUERY_MODEL, "query")

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, SOLAR_EMBEDDING_PASSAGE_MODEL, "passage")

    def _embed(self, texts: list[str], model: str, input_type: str) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if not SOLAR_EMBEDDING_API_KEY:
            raise RuntimeError(
                "STOREPILOT_SOLAR_EMBEDDING_API_KEY is required when "
                "STOREPILOT_EMBEDDING_PROVIDER=solar."
            )

        batches: list[np.ndarray] = []
        for start in range(0, len(texts), SOLAR_EMBEDDING_BATCH_SIZE):
            batch = texts[start:start + SOLAR_EMBEDDING_BATCH_SIZE]
            batches.append(self._embed_batch(batch, model, input_type))
        return np.vstack(batches).astype(np.float32, copy=False)

    def _embed_batch(self, texts: list[str], model: str, input_type: str) -> np.ndarray:
        _wait_for_rate_limit()
        started_at = perf_counter()
        request = urllib.request.Request(
            f"{SOLAR_EMBEDDING_API_BASE_URL}/embeddings",
            data=json.dumps({"model": model, "input": texts}, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {SOLAR_EMBEDDING_API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=SOLAR_EMBEDDING_TIMEOUT_SECONDS) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Solar embedding API request failed: HTTP {error.code} {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Solar embedding API request failed: {error.reason}") from error

        rows = sorted(body.get("data", []), key=lambda item: int(item.get("index", 0)))
        vectors = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise RuntimeError("Solar embedding API response count did not match request count.")
        if SOLAR_EMBEDDING_DIMENSIONS > 0 and vectors.shape[1] != SOLAR_EMBEDDING_DIMENSIONS:
            raise RuntimeError(
                "Solar embedding API response dimension did not match "
                f"STOREPILOT_SOLAR_EMBEDDING_DIMENSIONS: {vectors.shape[1]} != {SOLAR_EMBEDDING_DIMENSIONS}."
            )

        usage = body.get("usage", {})
        logger.info(
            "embedding_api_timing provider=solar input_type=%s model=%s inputs=%d dimensions=%d "
            "prompt_tokens=%s total_tokens=%s elapsed_ms=%.1f",
            input_type,
            model,
            len(texts),
            int(vectors.shape[1]),
            usage.get("prompt_tokens", "unknown"),
            usage.get("total_tokens", "unknown"),
            (perf_counter() - started_at) * 1000,
        )
        return _normalize(vectors)


def _wait_for_rate_limit() -> None:
    """Space Solar request starts across all threads in this process."""
    global _next_request_at

    if SOLAR_EMBEDDING_REQUESTS_PER_MINUTE <= 0:
        return

    interval_seconds = 60.0 / SOLAR_EMBEDDING_REQUESTS_PER_MINUTE
    with _rate_limit_lock:
        now = time.monotonic()
        wait_seconds = max(0.0, _next_request_at - now)
        if wait_seconds > 0:
            logger.info(
                "embedding_api_rate_limit provider=solar wait_ms=%.1f requests_per_minute=%d",
                wait_seconds * 1000,
                SOLAR_EMBEDDING_REQUESTS_PER_MINUTE,
            )
            time.sleep(wait_seconds)
        _next_request_at = max(_next_request_at, time.monotonic()) + interval_seconds


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
