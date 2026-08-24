import json

import numpy as np

from app.category_matcher.schemas import CategoryItem, PredictionCandidate
from app.category_matcher.config.settings import (
    CACHE_ROOT,
    CATEGORY_EMBEDDING_SEARCH_K,
)
from app.category_matcher.embedding.factory import get_embedding_provider


def embed(texts: list[str]) -> np.ndarray:
    return embed_queries(texts)


def embed_queries(texts: list[str]) -> np.ndarray:
    provider = get_embedding_provider()
    return provider.embed_queries(texts)


def embed_passages(texts: list[str]) -> np.ndarray:
    provider = get_embedding_provider()
    return provider.embed_passages(texts)


def rebuild_category_cache(version_id: int, categories: list[CategoryItem]) -> None:
    version_dir = category_cache_dir(version_id)
    version_dir.mkdir(parents=True, exist_ok=True)

    passages = [category_text(category) for category in categories]
    embeddings = embed_passages(passages)

    np.save(version_dir / "category_embeddings.npy", embeddings)
    provider = get_embedding_provider()
    (version_dir / "model.json").write_text(
        json.dumps(
            {
                "embeddingProvider": provider.provider_name,
                "modelName": provider.model_name,
                "dimension": int(embeddings.shape[1]),
            },
            ensure_ascii=False,
            indent=2,
        ),
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
    return CACHE_ROOT / get_embedding_provider().cache_key / f"version-{version_id}"


def category_text(category: CategoryItem) -> str:
    return category.searchText or category.fullPath
