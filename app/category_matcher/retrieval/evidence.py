from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from app.category_matcher.config.settings import (
    PRODUCT_DISTRIBUTION_TEMPERATURE,
    PRODUCT_MAX_PER_CATEGORY,
    PRODUCT_REPRESENTATIVE_K,
)
from app.category_matcher.product_memory.store import NaverCategoryLabel, ProductSearchHit
from app.category_matcher.schemas import (
    CategoryDistributionItem,
    SimilarProductItem,
)


@dataclass(frozen=True)
class ProductEvidence:
    similar_products: list[SimilarProductItem]
    distribution: list[CategoryDistributionItem]


def build_product_evidence(
    hits: list[ProductSearchHit],
) -> ProductEvidence:
    resolved: list[tuple[ProductSearchHit, NaverCategoryLabel]] = []

    for hit in hits:
        if len(hit.categories) == 1:
            resolved.append((hit, hit.categories[0]))

    distribution = _calculate_distribution(resolved)
    representatives = _select_representatives(resolved)
    return ProductEvidence(similar_products=representatives, distribution=distribution)


def _calculate_distribution(
    resolved: list[tuple[ProductSearchHit, NaverCategoryLabel]],
) -> list[CategoryDistributionItem]:
    if not resolved:
        return []

    max_score = max(hit.similarity for hit, _ in resolved)
    weights: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    max_scores: dict[int, float] = defaultdict(float)
    category_mapping: dict[int, NaverCategoryLabel] = {}

    for hit, mapping in resolved:
        weight = math.exp((hit.similarity - max_score) / PRODUCT_DISTRIBUTION_TEMPERATURE)
        weights[mapping.category_id] += weight
        counts[mapping.category_id] += 1
        max_scores[mapping.category_id] = max(max_scores[mapping.category_id], hit.similarity)
        category_mapping[mapping.category_id] = mapping

    total = sum(weights.values())
    return sorted(
        [
            CategoryDistributionItem(
                categoryId=category_id,
                categoryCode=category_mapping[category_id].category_code,
                fullPath=category_mapping[category_id].full_path,
                support=weight / total,
                exampleCount=counts[category_id],
                maxSimilarity=max_scores[category_id],
            )
            for category_id, weight in weights.items()
        ],
        key=lambda item: (item.support, item.maxSimilarity),
        reverse=True,
    )


def _select_representatives(
    resolved: list[tuple[ProductSearchHit, NaverCategoryLabel]],
) -> list[SimilarProductItem]:
    category_counts: dict[int, int] = defaultdict(int)
    selected: list[SimilarProductItem] = []

    for hit, mapping in resolved:
        if category_counts[mapping.category_id] >= PRODUCT_MAX_PER_CATEGORY:
            continue
        selected.append(
            SimilarProductItem(
                productName=hit.product_name,
                categoryId=mapping.category_id,
                categoryCode=mapping.category_code,
                fullPath=mapping.full_path,
                similarity=hit.similarity,
            )
        )
        category_counts[mapping.category_id] += 1
        if len(selected) >= PRODUCT_REPRESENTATIVE_K:
            break
    return selected
