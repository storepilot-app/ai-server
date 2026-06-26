import numpy as np

from app.category_matcher.rules.keywords import (
    BODY_KEYWORD_CATEGORY_TERMS,
    BODY_KEYWORDS_BY_TYPE,
    GUNPLA_STRONG_KEYWORDS,
)
from app.category_matcher.preprocess.query import detect_body_keywords
from app.category_matcher.config.settings import (
    BODY_KEYWORD_CATEGORY_BONUS,
    BODY_KEYWORD_EXACT_CATEGORY_BONUS,
    BODY_KEYWORD_NON_MATCH_PENALTY,
    GUNPLA_CATEGORY_BONUS,
)
from app.category_matcher.common.text import last_category_name, normalize_keyword_text


def apply_category_score_adjustments(product_name: str, row_scores: np.ndarray, categories: list[dict]) -> np.ndarray:
    body_keywords = detect_body_keywords(product_name)
    if body_keywords:
        return apply_body_keyword_category_bonus(product_name, row_scores, categories, body_keywords)
    return apply_gunpla_category_bonus(product_name, row_scores, categories)


def apply_gunpla_category_bonus(product_name: str, row_scores: np.ndarray, categories: list[dict]) -> np.ndarray:
    if not has_gunpla_keyword(product_name):
        return row_scores

    adjusted_scores = row_scores.copy()
    for index, category in enumerate(categories):
        if is_plamodel_category(category):
            adjusted_scores[index] += GUNPLA_CATEGORY_BONUS
    return adjusted_scores


def apply_body_keyword_category_bonus(
    product_name: str,
    row_scores: np.ndarray,
    categories: list[dict],
    body_keywords: list[str] | None = None,
) -> np.ndarray:
    body_keywords = body_keywords if body_keywords is not None else detect_body_keywords(product_name)
    if not body_keywords:
        return row_scores

    adjusted_scores = row_scores.copy()
    for index, category in enumerate(categories):
        full_path = category.get("fullPath", "")
        category_text_value = normalize_keyword_text(f"{full_path} {category.get('searchText', '')}")
        leaf_category = normalize_keyword_text(last_category_name(full_path))
        matched_body_category = False
        for body_keyword in body_keywords:
            category_terms = BODY_KEYWORD_CATEGORY_TERMS.get(body_keyword, [body_keyword])
            if any(normalize_keyword_text(term) in category_text_value for term in category_terms):
                adjusted_scores[index] += BODY_KEYWORD_CATEGORY_BONUS
                if any(normalize_keyword_text(term) == leaf_category for term in BODY_KEYWORDS_BY_TYPE.get(body_keyword, [body_keyword])):
                    adjusted_scores[index] += BODY_KEYWORD_EXACT_CATEGORY_BONUS
                matched_body_category = True
                break
        if not matched_body_category:
            adjusted_scores[index] -= BODY_KEYWORD_NON_MATCH_PENALTY
    return adjusted_scores


def has_gunpla_keyword(product_name: str) -> bool:
    normalized = normalize_keyword_text(product_name)
    if not normalized:
        return False
    return any(normalize_keyword_text(keyword) in normalized for keyword in GUNPLA_STRONG_KEYWORDS)


def is_plamodel_category(category: dict) -> bool:
    text = f"{category.get('fullPath', '')} {category.get('searchText', '')}"
    return "프라모델" in text
