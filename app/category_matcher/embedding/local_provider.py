import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.category_matcher.config.settings import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_USE_FP16,
    MODEL_NAME,
)


class LocalEmbeddingProvider:
    provider_name = "local"

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None

    @property
    def model_name(self) -> str:
        return MODEL_NAME

    @property
    def cache_key(self) -> str:
        from app.category_matcher.config.settings import MODEL_CACHE_KEY

        return MODEL_CACHE_KEY

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        vectors = self._get_model().encode(
            texts,
            normalize_embeddings=True,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            device = self._resolve_device()
            self._model = SentenceTransformer(MODEL_NAME, device=device)
            if device == "cuda" and EMBEDDING_USE_FP16:
                self._model.half()
        return self._model

    def _resolve_device(self) -> str:
        if EMBEDDING_DEVICE == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if EMBEDDING_DEVICE == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("STOREPILOT_EMBEDDING_DEVICE=cuda but CUDA is unavailable.")
        if EMBEDDING_DEVICE not in {"cpu", "cuda"}:
            raise ValueError(f"Unsupported embedding device: {EMBEDDING_DEVICE}")
        return EMBEDDING_DEVICE
