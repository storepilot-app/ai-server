from app.category_matcher.decision.policy import auto_accepted_category
from app.category_matcher.embedding.store import (
    embed,
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
    preprocess_embedding_query,
    preprocess_product_name,
)
from app.category_matcher.product_memory.store import ProductSearchHit, search_similar_products_by_vectors
from app.category_matcher.retrieval.evidence import build_product_evidence
from app.category_matcher.schemas import (
    CategoryDistributionItem,
    PredictionCandidate,
    PredictionItem,
    ProductItem,
)


def predict_categories(
    version_id: int,
    products: list[ProductItem],
) -> list[PredictionItem]:
    queries = [preprocess_embedding_query(product.productName) for product in products]
    query_embeddings = embed(queries)
    product_candidates = [ProductCandidates(product=product, candidates=[]) for product in products]
    product_hit_rows = search_similar_products_by_vectors(query_embeddings)
    resolved_items = [
        attach_product_evidence(item, hits)
        for item, hits in zip(product_candidates, product_hit_rows)
    ]
    completed_results: dict[int, PredictionItem] = {}
    ambiguous_items: list[ProductCandidates] = []

    for item in resolved_items:
        if not item.similar_products:
            completed_results[item.product.rowId] = prediction_without_similar_products(item)
            continue
        accepted = auto_accepted_category(item.category_distribution)
        if accepted is None:
            ambiguous_items.append(item)
            continue
        completed_results[item.product.rowId] = prediction_from_auto_accept(item, accepted)

    llm_selections = select_candidates_with_llm_batch(ambiguous_items)
    return [
        completed_results.get(item.product.rowId)
        or prediction_from_selection(
            item,
            llm_selections.get(
                item.product.rowId,
                fallback_llm_selection(item.candidates, "FAILED", "LLM batch response missing this rowId."),
            ),
        )
        for item in resolved_items
    ]


def prediction_without_similar_products(item: ProductCandidates) -> PredictionItem:
    return PredictionItem(
        rowId=item.product.rowId,
        categoryId=None,
        categoryCode=None,
        fullPath=None,
        score=0.0,
        candidates=[],
        llmUsed=False,
        llmSelectedCategory=None,
        llmStatus="NO_SIMILAR_PRODUCTS",
        llmStatusDetail=None,
        decisionSource="NO_SIMILAR_PRODUCTS",
        similarProducts=[],
        categoryDistribution=[],
    )


def attach_product_evidence(
    item: ProductCandidates,
    hits: list[ProductSearchHit],
) -> ProductCandidates:
    evidence = build_product_evidence(hits)
    return ProductCandidates(
        product=item.product,
        candidates=item.candidates,
        similar_products=evidence.similar_products,
        category_distribution=evidence.distribution,
    )


def prediction_from_auto_accept(
    item: ProductCandidates,
    accepted: CategoryDistributionItem,
) -> PredictionItem:
    selected = PredictionCandidate(
        categoryId=accepted.categoryId,
        categoryCode=accepted.categoryCode,
        fullPath=accepted.fullPath,
        score=accepted.maxSimilarity,
    )
    return PredictionItem(
        rowId=item.product.rowId,
        categoryId=selected.categoryId,
        categoryCode=selected.categoryCode,
        fullPath=selected.fullPath,
        score=selected.score,
        candidates=item.candidates,
        llmUsed=False,
        llmSelectedCategory=None,
        llmStatus="AUTO_SELECTED",
        llmStatusDetail="High-similarity product evidence agreed on one category.",
        decisionSource="PRODUCT_AUTO_ACCEPT",
        similarProducts=item.similar_products,
        categoryDistribution=item.category_distribution,
    )


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
            decisionSource="LLM_SIMILAR_PRODUCTS" if llm_selection.used else "NO_MATCH",
            similarProducts=item.similar_products,
            categoryDistribution=item.category_distribution,
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
        decisionSource="LLM_SIMILAR_PRODUCTS" if llm_selection.used else "NO_MATCH",
        similarProducts=item.similar_products,
        categoryDistribution=item.category_distribution,
    )
