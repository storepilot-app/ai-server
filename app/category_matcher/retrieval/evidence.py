from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from app.category_matcher.config.settings import (
    PRODUCT_DISTRIBUTION_TEMPERATURE,
    PRODUCT_MAX_PER_CATEGORY,
    PRODUCT_REPRESENTATIVE_K,
)
from app.category_matcher.product_memory.store import ProductSearchHit
from app.category_matcher.schemas import (
    CategoryDistributionItem,
    MyCategoryMappingItem,
    SimilarProductItem,
)


@dataclass(frozen=True)
class ProductEvidence:
    similar_products: list[SimilarProductItem]
    distribution: list[CategoryDistributionItem]


def build_product_evidence(
    hits: list[ProductSearchHit],
    mappings: list[MyCategoryMappingItem],
) -> ProductEvidence:
    mappings_by_code = {mapping.myCategoryCode: mapping for mapping in mappings}
    resolved: list[tuple[ProductSearchHit, MyCategoryMappingItem]] = []

    for hit in hits:
        mapped = {
            mapping.categoryId: mapping
            for code in hit.my_category_codes
            if (mapping := mappings_by_code.get(code)) is not None
        }
        if len(mapped) == 1:
            resolved.append((hit, next(iter(mapped.values()))))

    distribution = _calculate_distribution(resolved)
    representatives = _select_representatives(resolved)
    return ProductEvidence(similar_products=representatives, distribution=distribution)


def _calculate_distribution(
    resolved: list[tuple[ProductSearchHit, MyCategoryMappingItem]],
) -> list[CategoryDistributionItem]:
    if not resolved:
        return []

    max_score = max(hit.similarity for hit, _ in resolved)
    weights: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    max_scores: dict[int, float] = defaultdict(float)
    category_mapping: dict[int, MyCategoryMappingItem] = {}

    for hit, mapping in resolved:
        weight = math.exp((hit.similarity - max_score) / PRODUCT_DISTRIBUTION_TEMPERATURE)
        weights[mapping.categoryId] += weight
        counts[mapping.categoryId] += 1
        max_scores[mapping.categoryId] = max(max_scores[mapping.categoryId], hit.similarity)
        category_mapping[mapping.categoryId] = mapping

    total = sum(weights.values())
    return sorted(
        [
            CategoryDistributionItem(
                categoryId=category_id,
                categoryCode=category_mapping[category_id].categoryCode,
                fullPath=category_mapping[category_id].fullPath,
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
    resolved: list[tuple[ProductSearchHit, MyCategoryMappingItem]],
) -> list[SimilarProductItem]:
    category_counts: dict[int, int] = defaultdict(int)
    selected: list[SimilarProductItem] = []

    for hit, mapping in resolved:
        if category_counts[mapping.categoryId] >= PRODUCT_MAX_PER_CATEGORY:
            continue
        selected.append(
            SimilarProductItem(
                productName=hit.product_name,
                myCategoryCode=hit.my_category_codes[0],
                categoryId=mapping.categoryId,
                categoryCode=mapping.categoryCode,
                fullPath=mapping.fullPath,
                similarity=hit.similarity,
            )
        )
        category_counts[mapping.categoryId] += 1
        if len(selected) >= PRODUCT_REPRESENTATIVE_K:
            break
    return selected
