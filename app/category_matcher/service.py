import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from app.category_matcher.schemas import CategoryItem, PredictionItem, ProductItem

MODEL_NAME = "intfloat/multilingual-e5-small"
CACHE_ROOT = Path("ai-cache/categories")

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def rebuild_category_cache(version_id: int, categories: list[CategoryItem]) -> None:
    version_dir = CACHE_ROOT / f"version-{version_id}"
    version_dir.mkdir(parents=True, exist_ok=True)

    passages = [f"passage: {category.searchText or category.fullPath}" for category in categories]
    embeddings = embed(passages)

    np.save(version_dir / "category_embeddings.npy", embeddings)
    (version_dir / "category_meta.json").write_text(
        json.dumps([category.model_dump() for category in categories], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def predict_categories(version_id: int, products: list[ProductItem]) -> list[PredictionItem]:
    embeddings, categories = load_category_cache(version_id)
    if len(categories) == 0:
        return [
            PredictionItem(rowId=product.rowId, categoryId=None, categoryCode=None, fullPath=None, score=0.0)
            for product in products
        ]

    queries = [f"query: {preprocess_product_name(product.productName)}" for product in products]
    query_embeddings = embed(queries)
    scores = query_embeddings @ embeddings.T
    top_indexes = scores.argmax(axis=1)

    results: list[PredictionItem] = []
    for product, top_index, row_scores in zip(products, top_indexes, scores):
        category = categories[int(top_index)]
        results.append(
            PredictionItem(
                rowId=product.rowId,
                categoryId=category["categoryId"],
                categoryCode=category["categoryCode"],
                fullPath=category["fullPath"],
                score=float(row_scores[int(top_index)]),
            )
        )
    return results


def embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    vectors = get_model().encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def load_category_cache(version_id: int) -> tuple[np.ndarray, list[dict]]:
    version_dir = CACHE_ROOT / f"version-{version_id}"
    embeddings_path = version_dir / "category_embeddings.npy"
    meta_path = version_dir / "category_meta.json"
    if not embeddings_path.exists() or not meta_path.exists():
        return np.empty((0, 384), dtype=np.float32), []

    embeddings = np.load(embeddings_path)
    categories = json.loads(meta_path.read_text(encoding="utf-8"))
    return embeddings, categories


def preprocess_product_name(product_name: str) -> str:
    text = product_name or ""
    text = re.sub(r"[\[\](){}]", " ", text)
    text = re.sub(r"[_/|,]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
