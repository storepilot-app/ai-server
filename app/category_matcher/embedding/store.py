import json

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.category_matcher.schemas import CategoryItem, PredictionCandidate
from app.category_matcher.config.settings import (
    CACHE_ROOT,
    CATEGORY_EMBEDDING_SEARCH_K,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_USE_FP16,
    MODEL_CACHE_KEY,
    MODEL_NAME,
)


_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        device = _resolve_device()
        _model = SentenceTransformer(MODEL_NAME, device=device)
        if device == "cuda" and EMBEDDING_USE_FP16:
            _model.half()
    return _model


def embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    vectors = get_model().encode(
        texts,
        normalize_embeddings=True,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def _resolve_device() -> str:
    if EMBEDDING_DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if EMBEDDING_DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("STOREPILOT_EMBEDDING_DEVICE=cuda but CUDA is unavailable.")
    if EMBEDDING_DEVICE not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported embedding device: {EMBEDDING_DEVICE}")
    return EMBEDDING_DEVICE


def rebuild_category_cache(version_id: int, categories: list[CategoryItem]) -> None:
    version_dir = category_cache_dir(version_id)
    version_dir.mkdir(parents=True, exist_ok=True)

    passages = [category_text(category) for category in categories]
    embeddings = embed(passages)

    np.save(version_dir / "category_embeddings.npy", embeddings)
    (version_dir / "model.json").write_text(
        json.dumps({"modelName": MODEL_NAME, "dimension": int(embeddings.shape[1])}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (version_dir / "category_meta.json").write_text(
        json.dumps([category.model_dump() for category in categories], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_category_cache(version_id: int) -> tuple[np.ndarray, list[dict]]:
    version_dir = category_cache_dir(version_id)
    embeddings_path = version_dir / "category_embeddings.npy"
    meta_path = version_dir / "category_meta.json"
    if not embeddings_path.exists() or not meta_path.exists():
        return np.empty((0, 0), dtype=np.float32), []

    embeddings = np.load(embeddings_path)
    categories = json.loads(meta_path.read_text(encoding="utf-8"))
    return embeddings, categories


def load_category_metadata(version_id: int) -> list[dict]:
    meta_path = category_cache_dir(version_id) / "category_meta.json"
    if not meta_path.exists():
        return []
    return json.loads(meta_path.read_text(encoding="utf-8"))


def search_category_candidates_by_vectors(
    version_id: int,
    query_vectors: np.ndarray,
) -> list[list[PredictionCandidate]]:
    if query_vectors.size == 0:
        return []

    category_embeddings, categories = load_category_cache(version_id)
    if category_embeddings.size == 0 or not categories:
        return [[] for _ in range(len(query_vectors))]
    if query_vectors.shape[1] != category_embeddings.shape[1]:
        return [[] for _ in range(len(query_vectors))]

    scores = query_vectors @ category_embeddings.T
    limit = min(max(1, CATEGORY_EMBEDDING_SEARCH_K), len(categories))
    top_indices = np.argpartition(scores, -limit, axis=1)[:, -limit:]
    results: list[list[PredictionCandidate]] = []

    for row_index, indices in enumerate(top_indices):
        sorted_indices = sorted(indices, key=lambda index: float(scores[row_index, index]), reverse=True)
        results.append([
            PredictionCandidate(
                categoryId=int(categories[index]["categoryId"]),
                categoryCode=str(categories[index]["categoryCode"]),
                fullPath=str(categories[index]["fullPath"]),
                score=float(scores[row_index, index]),
            )
            for index in sorted_indices
        ])
    return results


def category_cache_dir(version_id: int):
    return CACHE_ROOT / MODEL_CACHE_KEY / f"version-{version_id}"


def category_text(category: CategoryItem) -> str:
    return category.searchText or category.fullPath
