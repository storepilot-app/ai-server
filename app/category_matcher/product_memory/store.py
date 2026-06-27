from __future__ import annotations

import hashlib
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
    MODEL_NAME,
    PRODUCT_CACHE_ROOT,
    PRODUCT_DUPLICATE_THRESHOLD,
    PRODUCT_HNSW_EF_CONSTRUCTION,
    PRODUCT_HNSW_EF_SEARCH,
    PRODUCT_HNSW_M,
    PRODUCT_INDEX_TYPE,
    PRODUCT_SEARCH_K,
)
from app.category_matcher.embedding.store import embed
from app.category_matcher.preprocess.query import preprocess_embedding_query
from app.category_matcher.product_memory.excel import read_product_rows


@dataclass(frozen=True)
class HistoricalProduct:
    product_name: str
    normalized_title: str
    my_category_codes: tuple[str, ...]


@dataclass(frozen=True)
class ProductSearchHit:
    product_name: str
    my_category_codes: tuple[str, ...]
    similarity: float


@dataclass(frozen=True)
class ProductIndexBuildResult:
    valid_row_count: int
    indexed_product_count: int
    duplicate_row_count: int
    conflicting_title_count: int


@dataclass
class LoadedProductIndex:
    index: faiss.Index
    products: list[HistoricalProduct]


_loaded_indexes: dict[str, LoadedProductIndex] = {}
_lock = threading.RLock()


def normalize_product_title(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", value.casefold())


def rebuild_product_index(
    user_key: str,
    sources: list[str | Path | BinaryIO],
) -> ProductIndexBuildResult:
    if not user_key.strip():
        raise ValueError("User key is required.")
    labels_by_title: dict[str, set[str]] = {}
    display_titles: dict[str, str] = {}
    valid_rows = 0

    for source in sources:
        if hasattr(source, "seek"):
            source.seek(0)
        for product_name, my_category_code in read_product_rows(source):
            normalized = normalize_product_title(product_name)
            if not normalized:
                continue
            valid_rows += 1
            display_titles.setdefault(normalized, product_name.strip())
            labels_by_title.setdefault(normalized, set()).add(my_category_code.strip())

    products = [
        HistoricalProduct(
            product_name=display_titles[normalized],
            normalized_title=normalized,
            my_category_codes=tuple(sorted(category_codes)),
        )
        for normalized, category_codes in labels_by_title.items()
    ]
    vectors = _embed_products(products)
    index = _new_index(int(vectors.shape[1]))
    index.add(vectors)

    with _lock:
        _save_index(user_key, index, products)
        _loaded_indexes[_user_cache_key(user_key)] = LoadedProductIndex(index=index, products=products)

    return ProductIndexBuildResult(
        valid_row_count=valid_rows,
        indexed_product_count=len(products),
        duplicate_row_count=valid_rows - len(products),
        conflicting_title_count=sum(len(product.my_category_codes) > 1 for product in products),
    )


def search_similar_products(user_key: str | None, product_name: str) -> list[ProductSearchHit]:
    if not user_key or not product_name.strip():
        return []

    query = embed([preprocess_embedding_query(product_name)])
    return search_similar_products_by_vector(user_key, query[0])


def search_similar_products_by_vector(
    user_key: str | None,
    query_vector: np.ndarray,
) -> list[ProductSearchHit]:
    results = search_similar_products_by_vectors(user_key, np.asarray(query_vector).reshape(1, -1))
    return results[0] if results else []


def search_similar_products_by_vectors(
    user_key: str | None,
    query_vectors: np.ndarray,
) -> list[list[ProductSearchHit]]:
    if not user_key:
        return [[] for _ in range(len(query_vectors))]

    loaded = _load_index(user_key)
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
    selected: list[tuple[int, np.ndarray]] = []
    hits: list[ProductSearchHit] = []

    for item_index, score in zip(indexes, scores):
        if item_index < 0:
            continue
        candidate_vector = loaded.index.reconstruct(int(item_index))
        if any(float(candidate_vector @ selected_vector) >= PRODUCT_DUPLICATE_THRESHOLD for _, selected_vector in selected):
            continue

        product = loaded.products[int(item_index)]
        selected.append((int(item_index), candidate_vector))
        hits.append(
            ProductSearchHit(
                product_name=product.product_name,
                my_category_codes=product.my_category_codes,
                similarity=float(np.clip(score, -1.0, 1.0)),
            )
        )
        if len(hits) >= PRODUCT_SEARCH_K:
            break
    return hits


def add_product_feedback(user_key: str, product_name: str, my_category_code: str) -> int:
    if not user_key.strip():
        raise ValueError("User key is required.")
    normalized = normalize_product_title(product_name)
    if not normalized or not my_category_code.strip():
        raise ValueError("Product name and my category code are required.")

    with _lock:
        loaded = _load_index(user_key)
        if loaded is None:
            vector = embed([preprocess_embedding_query(product_name)])
            index = _new_index(int(vector.shape[1]))
            index.add(vector)
            products = [HistoricalProduct(product_name.strip(), normalized, (my_category_code.strip(),))]
            _save_index(user_key, index, products)
            _loaded_indexes[_user_cache_key(user_key)] = LoadedProductIndex(index, products)
            return 1

        for index, product in enumerate(loaded.products):
            if product.normalized_title != normalized:
                continue
            loaded.products[index] = HistoricalProduct(
                product_name.strip(),
                normalized,
                (my_category_code.strip(),),
            )
            _save_index(user_key, loaded.index, loaded.products)
            return len(loaded.products)

        vector = embed([preprocess_embedding_query(product_name)])
        loaded.index.add(vector)
        loaded.products.append(HistoricalProduct(product_name.strip(), normalized, (my_category_code.strip(),)))
        _save_index(user_key, loaded.index, loaded.products)
        return len(loaded.products)


def _embed_products(products: list[HistoricalProduct]) -> np.ndarray:
    if not products:
        raise ValueError("No valid product rows were found.")

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


def _load_index(user_key: str) -> LoadedProductIndex | None:
    cache_key = _user_cache_key(user_key)
    with _lock:
        if cache_key in _loaded_indexes:
            return _loaded_indexes[cache_key]

        directory = _product_cache_dir(user_key)
        index_path = directory / "products.faiss"
        metadata_path = directory / "products.json"
        if not index_path.exists() or not metadata_path.exists():
            return None

        index = faiss.read_index(str(index_path))
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if payload.get("modelName") != MODEL_NAME or payload.get("indexType", "flat") != PRODUCT_INDEX_TYPE:
            return None
        if PRODUCT_INDEX_TYPE == "hnsw":
            index.hnsw.efSearch = PRODUCT_HNSW_EF_SEARCH
        products = [
            HistoricalProduct(
                product_name=item["productName"],
                normalized_title=item["normalizedTitle"],
                my_category_codes=tuple(item["myCategoryCodes"]),
            )
            for item in payload["products"]
        ]
        if index.ntotal != len(products):
            return None
        loaded = LoadedProductIndex(index=index, products=products)
        _loaded_indexes[cache_key] = loaded
        return loaded


def _save_index(user_key: str, index: faiss.Index, products: list[HistoricalProduct]) -> None:
    directory = _product_cache_dir(user_key)
    directory.mkdir(parents=True, exist_ok=True)
    index_temp = directory / "products.faiss.tmp"
    metadata_temp = directory / "products.json.tmp"

    faiss.write_index(index, str(index_temp))
    metadata_temp.write_text(
        json.dumps(
            {
                "modelName": MODEL_NAME,
                "indexType": PRODUCT_INDEX_TYPE,
                "products": [
                    {
                        "productName": product.product_name,
                        "normalizedTitle": product.normalized_title,
                        "myCategoryCodes": list(product.my_category_codes),
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


def _product_cache_dir(user_key: str) -> Path:
    return PRODUCT_CACHE_ROOT / MODEL_CACHE_KEY / _user_cache_key(user_key)


def _user_cache_key(user_key: str) -> str:
    return hashlib.sha256(user_key.strip().encode("utf-8")).hexdigest()[:20]


def _new_index(dimension: int) -> faiss.Index:
    if PRODUCT_INDEX_TYPE == "flat":
        return faiss.IndexFlatIP(dimension)
    if PRODUCT_INDEX_TYPE == "hnsw":
        index = faiss.IndexHNSWFlat(dimension, PRODUCT_HNSW_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = PRODUCT_HNSW_EF_CONSTRUCTION
        index.hnsw.efSearch = PRODUCT_HNSW_EF_SEARCH
        return index
    raise ValueError(f"Unsupported product index type: {PRODUCT_INDEX_TYPE}")
