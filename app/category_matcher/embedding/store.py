import json

import numpy as np
from sentence_transformers import SentenceTransformer

from app.category_matcher.schemas import CategoryItem
from app.category_matcher.config.settings import CACHE_ROOT, MODEL_CACHE_KEY, MODEL_NAME


_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    vectors = get_model().encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


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


def category_cache_dir(version_id: int):
    return CACHE_ROOT / MODEL_CACHE_KEY / f"version-{version_id}"


def category_text(category: CategoryItem) -> str:
    return category.searchText or category.fullPath
