import json
import logging
import re
from time import perf_counter
import urllib.error
import urllib.request

import numpy as np

from app.category_matcher.config.settings import (
    EMBEDDING_API_BASE_URL,
    EMBEDDING_API_BATCH_SIZE,
    EMBEDDING_API_DIMENSIONS,
    EMBEDDING_API_KEY,
    EMBEDDING_API_MODEL,
    EMBEDDING_API_TIMEOUT_SECONDS,
)


logger = logging.getLogger("uvicorn.error").getChild("storepilot.embedding_api")


class OpenAIEmbeddingProvider:
    provider_name = "openai"
    is_asymmetric = False

    @property
    def model_name(self) -> str:
        return EMBEDDING_API_MODEL

    @property
    def cache_key(self) -> str:
        value = f"{self.provider_name}-{self.model_name}-{EMBEDDING_API_DIMENSIONS}"
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower()

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if not EMBEDDING_API_KEY:
            raise RuntimeError("STOREPILOT_EMBEDDING_API_KEY is required when STOREPILOT_EMBEDDING_PROVIDER=openai.")

        batches: list[np.ndarray] = []
        batch_size = max(1, EMBEDDING_API_BATCH_SIZE)
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            batches.append(self._embed_batch(batch))
        return np.vstack(batches).astype(np.float32, copy=False)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        started_at = perf_counter()
        payload: dict[str, object] = {
            "model": EMBEDDING_API_MODEL,
            "input": texts,
        }
        if EMBEDDING_API_DIMENSIONS > 0:
            payload["dimensions"] = EMBEDDING_API_DIMENSIONS

        request = urllib.request.Request(
            f"{EMBEDDING_API_BASE_URL}/embeddings",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {EMBEDDING_API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=EMBEDDING_API_TIMEOUT_SECONDS) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Embedding API request failed: HTTP {error.code} {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Embedding API request failed: {error.reason}") from error

        rows = sorted(body.get("data", []), key=lambda item: int(item.get("index", 0)))
        vectors = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
        if vectors.shape[0] != len(texts):
            raise RuntimeError("Embedding API response count did not match request count.")
        usage = body.get("usage", {})
        logger.info(
            "embedding_api_timing model=%s inputs=%d dimensions=%d prompt_tokens=%s total_tokens=%s elapsed_ms=%.1f",
            EMBEDDING_API_MODEL,
            len(texts),
            int(vectors.shape[1]),
            usage.get("prompt_tokens", "unknown"),
            usage.get("total_tokens", "unknown"),
            (perf_counter() - started_at) * 1000,
        )
        return _normalize(vectors)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
