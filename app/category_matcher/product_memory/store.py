from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import faiss
import numpy as np

from app.category_matcher.config.settings import (
    MODEL_CACHE_KEY,
    PRODUCT_CACHE_ROOT,
    PRODUCT_DUPLICATE_THRESHOLD,
    PRODUCT_HNSW_EF_CONSTRUCTION,
    PRODUCT_HNSW_EF_SEARCH,
    PRODUCT_HNSW_M,
    PRODUCT_INDEX_TYPE,
    PRODUCT_SEARCH_K,
)
from app.category_matcher.embedding.factory import get_embedding_provider
from app.category_matcher.embedding.store import embed
from app.category_matcher.preprocess.query import preprocess_embedding_query
from app.category_matcher.product_memory.excel import read_product_rows


PRODUCT_INDEX_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class NaverCategoryLabel:
    category_id: int
    category_code: str
    full_path: str


@dataclass(frozen=True)
class HistoricalProduct:
    product_name: str
    normalized_title: str
    categories: tuple[NaverCategoryLabel, ...]


@dataclass(frozen=True)
class ProductSearchHit:
    product_name: str
    categories: tuple[NaverCategoryLabel, ...]
    similarity: float


@dataclass(frozen=True)
class ProductIndexBuildResult:
    source_row_count: int
    valid_row_count: int
    unmapped_row_count: int
    indexed_product_count: int
    duplicate_row_count: int
    conflicting_title_count: int


@dataclass
class LoadedProductIndex:
    index: faiss.Index
    products: list[HistoricalProduct]


_loaded_index: LoadedProductIndex | None = None
_lock = threading.RLock()


def normalize_product_title(value: str) -> str:
    return re.sub(r"[^0-9a-z\uac00-\ud7a3]", "", value.casefold())


def rebuild_product_index(
    sources: list[str | Path | BinaryIO],
    mappings: dict[str, NaverCategoryLabel],
) -> ProductIndexBuildResult:
    global _loaded_index
    if not mappings:
        raise ValueError("At least one my-category to Naver-category mapping is required.")

    labels_by_title: dict[str, set[NaverCategoryLabel]] = {}
    display_titles: dict[str, str] = {}
    source_rows = 0
    valid_rows = 0

    for source in sources:
        if hasattr(source, "seek"):
            source.seek(0)
        for product_name, my_category_code in read_product_rows(source):
            source_rows += 1
            category = mappings.get(my_category_code.strip())
            if category is None:
                continue
            normalized = normalize_product_title(product_name)
            if not normalized:
                continue
            valid_rows += 1
            display_titles.setdefault(normalized, product_name.strip())
            labels_by_title.setdefault(normalized, set()).add(category)

    products = [
        HistoricalProduct(
            product_name=display_titles[normalized],
            normalized_title=normalized,
            categories=tuple(sorted(categories, key=lambda item: item.category_code)),
        )
        for normalized, categories in labels_by_title.items()
    ]
    vectors = _embed_products(products)
    index = _new_index(int(vectors.shape[1]))
    index.add(vectors)

    with _lock:
        _save_index(index, products)
        _loaded_index = LoadedProductIndex(index=index, products=products)

    return ProductIndexBuildResult(
        source_row_count=source_rows,
        valid_row_count=valid_rows,
        unmapped_row_count=source_rows - valid_rows,
        indexed_product_count=len(products),
        duplicate_row_count=valid_rows - len(products),
        conflicting_title_count=sum(len(product.categories) > 1 for product in products),
    )


def search_similar_products(product_name: str) -> list[ProductSearchHit]:
    if not product_name.strip():
        return []
    query = embed([preprocess_embedding_query(product_name)])
    return search_similar_products_by_vector(query[0])


def search_similar_products_by_vector(query_vector: np.ndarray) -> list[ProductSearchHit]:
    results = search_similar_products_by_vectors(np.asarray(query_vector).reshape(1, -1))
    return results[0] if results else []


def search_similar_products_by_vectors(query_vectors: np.ndarray) -> list[list[ProductSearchHit]]:
    loaded = _load_index()
    if loaded is None or loaded.index.ntotal == 0:
        return [[] for _ in range(len(query_vectors))]

    queries = np.ascontiguousarray(query_vectors, dtype=np.float32)
    if queries.ndim != 2 or queries.shape[1] != loaded.index.d:
        return [[] for _ in range(len(query_vectors))]
    search_count = min(loaded.index.ntotal, max(PRODUCT_SEARCH_K * 3, PRODUCT_SEARCH_K))
    score_rows, index_rows = loaded.index.search(queries, search_count)
    return [
        _collapse_search_results(loaded, scores, indexes)
        for scores, indexes in zip(score_rows, index_rows)
    ]


def _collapse_search_results(
    loaded: LoadedProductIndex,
    scores: np.ndarray,
    indexes: np.ndarray,
) -> list[ProductSearchHit]:
    selected_vectors: list[np.ndarray] = []
    hits: list[ProductSearchHit] = []

    for item_index, score in zip(indexes, scores):
        if item_index < 0:
            continue
        candidate_vector = loaded.index.reconstruct(int(item_index))
        if any(float(candidate_vector @ selected) >= PRODUCT_DUPLICATE_THRESHOLD for selected in selected_vectors):
            continue

        product = loaded.products[int(item_index)]
        selected_vectors.append(candidate_vector)
        hits.append(
            ProductSearchHit(
                product_name=product.product_name,
                categories=product.categories,
                similarity=float(np.clip(score, -1.0, 1.0)),
            )
        )
        if len(hits) >= PRODUCT_SEARCH_K:
            break
    return hits


def add_product_feedback(product_name: str, category: NaverCategoryLabel) -> int:
    global _loaded_index
    normalized = normalize_product_title(product_name)
    if not normalized or not category.category_code.strip():
        raise ValueError("Product name and Naver category are required.")

    with _lock:
        loaded = _load_index()
        if loaded is None:
            vector = embed([preprocess_embedding_query(product_name)])
            index = _new_index(int(vector.shape[1]))
            index.add(vector)
            products = [HistoricalProduct(product_name.strip(), normalized, (category,))]
            _save_index(index, products)
            _loaded_index = LoadedProductIndex(index, products)
            return 1

        for index, product in enumerate(loaded.products):
            if product.normalized_title != normalized:
                continue
            loaded.products[index] = HistoricalProduct(product_name.strip(), normalized, (category,))
            _save_index(loaded.index, loaded.products)
            return len(loaded.products)

        vector = embed([preprocess_embedding_query(product_name)])
        loaded.index.add(vector)
        loaded.products.append(HistoricalProduct(product_name.strip(), normalized, (category,)))
        _save_index(loaded.index, loaded.products)
        return len(loaded.products)


def _embed_products(products: list[HistoricalProduct]) -> np.ndarray:
    if not products:
        raise ValueError("No product rows had a valid Naver category mapping.")

    batches: list[np.ndarray] = []
    for start in range(0, len(products), 512):
        batch = products[start:start + 512]
        batches.append(embed([preprocess_embedding_query(product.product_name) for product in batch]))
        completed = min(start + len(batch), len(products))
        print(
            f"Product embedding progress: {completed:,}/{len(products):,} "
            f"({completed * 100 / len(products):.1f}%)",
            flush=True,
        )
    return np.vstack(batches).astype(np.float32, copy=False)


def _load_index() -> LoadedProductIndex | None:
    global _loaded_index
    with _lock:
        if _loaded_index is not None:
            return _loaded_index

        directory = _product_cache_dir()
        index_path = directory / "products.faiss"
        metadata_path = directory / "products.json"
        if not index_path.exists() or not metadata_path.exists():
            return None

        index = faiss.read_index(str(index_path))
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        provider = get_embedding_provider()
        if (
            payload.get("schemaVersion") != PRODUCT_INDEX_SCHEMA_VERSION
            or payload.get("embeddingProvider", "local") != provider.provider_name
            or payload.get("modelName") != provider.model_name
            or payload.get("indexType", "flat") != PRODUCT_INDEX_TYPE
        ):
            return None
        if PRODUCT_INDEX_TYPE == "hnsw":
            index.hnsw.efSearch = PRODUCT_HNSW_EF_SEARCH
        products = [
            HistoricalProduct(
                product_name=item["productName"],
                normalized_title=item["normalizedTitle"],
                categories=tuple(
                    NaverCategoryLabel(
                        category_id=category["categoryId"],
                        category_code=category["categoryCode"],
                        full_path=category["fullPath"],
                    )
                    for category in item["naverCategories"]
                ),
            )
            for item in payload["products"]
        ]
        if index.ntotal != len(products):
            return None
        _loaded_index = LoadedProductIndex(index=index, products=products)
        return _loaded_index


def _save_index(index: faiss.Index, products: list[HistoricalProduct]) -> None:
    directory = _product_cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    index_temp = directory / "products.faiss.tmp"
    metadata_temp = directory / "products.json.tmp"

    faiss.write_index(index, str(index_temp))
    provider = get_embedding_provider()
    metadata_temp.write_text(
        json.dumps(
            {
                "schemaVersion": PRODUCT_INDEX_SCHEMA_VERSION,
                "embeddingProvider": provider.provider_name,
                "modelName": provider.model_name,
                "indexType": PRODUCT_INDEX_TYPE,
                "products": [
                    {
                        "productName": product.product_name,
                        "normalizedTitle": product.normalized_title,
                        "naverCategories": [
                            {
                                "categoryId": category.category_id,
                                "categoryCode": category.category_code,
                                "fullPath": category.full_path,
                            }
                            for category in product.categories
                        ],
                    }
                    for product in products
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    os.replace(index_temp, directory / "products.faiss")
    os.replace(metadata_temp, directory / "products.json")


def _product_cache_dir() -> Path:
    return PRODUCT_CACHE_ROOT / get_embedding_provider().cache_key / "shared"


def _new_index(dimension: int) -> faiss.Index:
    if PRODUCT_INDEX_TYPE == "flat":
        return faiss.IndexFlatIP(dimension)
    if PRODUCT_INDEX_TYPE == "hnsw":
        index = faiss.IndexHNSWFlat(dimension, PRODUCT_HNSW_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = PRODUCT_HNSW_EF_CONSTRUCTION
        index.hnsw.efSearch = PRODUCT_HNSW_EF_SEARCH
        return index
    raise ValueError(f"Unsupported product index type: {PRODUCT_INDEX_TYPE}")
