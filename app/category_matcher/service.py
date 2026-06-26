import numpy as np

from app.category_matcher.embedding.store import (
    category_cache_dir,
    category_text,
    embed,
    get_model,
    load_category_cache,
    rebuild_category_cache,
)
from app.category_matcher.llm.judge import (
    LlmSelection,
    ProductCandidates,
    fallback_llm_selection,
    parse_llm_json,
    select_candidate_with_llm,
    select_candidates_with_llm_batch,
    selection_from_llm_decision,
)
from app.category_matcher.preprocess.query import (
    apply_body_keyword_focus,
    apply_tail_token_weight,
    detect_body_keywords,
    emphasize_noun_like_terms,
    extract_noun_focused_terms,
    get_kiwi,
    preprocess_embedding_query,
    preprocess_product_name,
)
from app.category_matcher.schemas import PredictionCandidate, PredictionItem, ProductItem
from app.category_matcher.score.adjuster import (
    apply_body_keyword_category_bonus,
    apply_category_score_adjustments,
    apply_gunpla_category_bonus,
    has_gunpla_keyword,
)


def predict_categories(version_id: int, products: list[ProductItem]) -> list[PredictionItem]:
    embeddings, categories = load_category_cache(version_id)
    if len(categories) == 0:
        return [empty_prediction(product) for product in products]

    queries = [preprocess_embedding_query(product.productName) for product in products]
    query_embeddings = embed(queries)
    if query_embeddings.shape[1] != embeddings.shape[1]:
        return [empty_prediction(product) for product in products]

    scores = query_embeddings @ embeddings.T
    product_candidates = build_product_candidates(products, categories, scores)
    llm_selections = select_candidates_with_llm_batch(product_candidates)
    return [
        prediction_from_selection(
            item,
            llm_selections.get(
                item.product.rowId,
                fallback_llm_selection(item.candidates, "FAILED", "LLM batch response missing this rowId."),
            ),
        )
        for item in product_candidates
    ]


def empty_prediction(product: ProductItem) -> PredictionItem:
    return PredictionItem(
        rowId=product.rowId,
        categoryId=None,
        categoryCode=None,
        fullPath=None,
        score=0.0,
        candidates=[],
    )


def build_product_candidates(
    products: list[ProductItem],
    categories: list[dict],
    scores: np.ndarray,
) -> list[ProductCandidates]:
    product_candidates: list[ProductCandidates] = []
    for product, row_scores in zip(products, scores):
        adjusted_scores = apply_category_score_adjustments(product.productName, row_scores, categories)
        top_indexes = np.argsort(adjusted_scores)[::-1][:10]
        candidates = [
            PredictionCandidate(
                categoryId=candidate["categoryId"],
                categoryCode=candidate["categoryCode"],
                fullPath=candidate["fullPath"],
                score=float(adjusted_scores[int(candidate_index)]),
            )
            for candidate_index in top_indexes
            for candidate in [categories[int(candidate_index)]]
        ]
        product_candidates.append(ProductCandidates(product=product, candidates=candidates))
    return product_candidates


def prediction_from_selection(item: ProductCandidates, llm_selection: LlmSelection) -> PredictionItem:
    selected_candidate = llm_selection.selected_candidate
    if selected_candidate is None:
        return PredictionItem(
            rowId=item.product.rowId,
            categoryId=None,
            categoryCode=None,
            fullPath=None,
            score=0.0,
            candidates=item.candidates,
            llmUsed=llm_selection.used,
            llmSelectedCategory=None,
            llmStatus=llm_selection.status,
            llmStatusDetail=llm_selection.detail,
        )

    return PredictionItem(
        rowId=item.product.rowId,
        categoryId=selected_candidate.categoryId,
        categoryCode=selected_candidate.categoryCode,
        fullPath=selected_candidate.fullPath,
        score=selected_candidate.score,
        candidates=item.candidates,
        llmUsed=llm_selection.used,
        llmSelectedCategory=selected_candidate.fullPath if llm_selection.used else None,
        llmStatus=llm_selection.status,
        llmStatusDetail=llm_selection.detail,
    )
