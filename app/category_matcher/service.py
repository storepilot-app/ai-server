import logging
from time import perf_counter

from app.category_matcher.config.settings import CATEGORY_EMBEDDING_CANDIDATE_K, PRODUCT_REPRESENTATIVE_K
from app.category_matcher.decision.policy import auto_accepted_category
from app.category_matcher.embedding.store import (
    embed,
    load_category_metadata,
    rebuild_category_cache,
    search_category_candidates_by_vectors,
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
from app.category_matcher.rules.category_alias import load_category_aliases, resolve_category_alias
from app.category_matcher.schemas import (
    CategoryDistributionItem,
    PredictionCandidate,
    PredictionItem,
    ProductItem,
)


logger = logging.getLogger("uvicorn.error").getChild("storepilot.category_matcher")


def predict_categories(
    version_id: int,
    products: list[ProductItem],
) -> list[PredictionItem]:
    request_started_at = perf_counter()

    preprocess_started_at = perf_counter()
    queries = [preprocess_embedding_query(product.productName) for product in products]
    preprocess_ms = elapsed_ms(preprocess_started_at)

    embedding_started_at = perf_counter()
    query_embeddings = embed(queries)
    embedding_ms = elapsed_ms(embedding_started_at)

    product_candidates = [ProductCandidates(product=product, candidates=[]) for product in products]
    search_started_at = perf_counter()
    product_hit_rows = search_similar_products_by_vectors(query_embeddings)
    search_ms = elapsed_ms(search_started_at)

    category_search_started_at = perf_counter()
    category_candidate_rows = search_category_candidates_by_vectors(version_id, query_embeddings)
    category_search_ms = elapsed_ms(category_search_started_at)
    category_metadata = load_category_metadata(version_id)
    category_aliases = load_category_aliases()
    alias_candidates = {
        product.rowId: candidate
        for product in products
        if (
            candidate := resolve_category_alias(
                product.productName,
                category_metadata,
                category_aliases,
            )
        ) is not None
    }

    evidence_started_at = perf_counter()
    resolved_items = [
        attach_product_evidence(item, hits, category_candidates)
        for item, hits, category_candidates in zip(
            product_candidates,
            product_hit_rows,
            category_candidate_rows,
        )
    ]
    evidence_ms = elapsed_ms(evidence_started_at)

    decision_started_at = perf_counter()
    completed_results: dict[int, PredictionItem] = {}
    ambiguous_items: list[ProductCandidates] = []
    no_similar_product_count = 0
    auto_selected_count = 0

    for item in resolved_items:
        alias_candidate = alias_candidates.get(item.product.rowId)
        if alias_candidate is not None:
            completed_results[item.product.rowId] = prediction_from_alias(item, alias_candidate)
            auto_selected_count += 1
            continue
        if not item.similar_products:
            if item.candidates:
                ambiguous_items.append(item)
            else:
                completed_results[item.product.rowId] = prediction_without_similar_products(item)
                no_similar_product_count += 1
            continue
        accepted = auto_accepted_category(item.category_distribution)
        if accepted is None:
            ambiguous_items.append(item)
            continue
        completed_results[item.product.rowId] = prediction_from_auto_accept(item, accepted)
        auto_selected_count += 1
    decision_ms = elapsed_ms(decision_started_at)

    llm_started_at = perf_counter()
    llm_selections = select_candidates_with_llm_batch(ambiguous_items)
    llm_ms = elapsed_ms(llm_started_at)

    response_started_at = perf_counter()
    results = [
        completed_results.get(item.product.rowId)
        or prediction_from_selection(
            item,
            llm_selections.get(
                item.product.rowId,
                fallback_llm_selection(
                    item.selection_candidates(), "FAILED", "LLM batch response missing this rowId."
                ),
            ),
        )
        for item in resolved_items
    ]
    response_ms = elapsed_ms(response_started_at)
    logger.info(
        "category_predict_timing version_id=%s products=%d no_similar=%d auto_selected=%d "
        "llm_items=%d preprocess_ms=%.1f embedding_ms=%.1f faiss_ms=%.1f category_search_ms=%.1f evidence_ms=%.1f "
        "decision_ms=%.1f llm_ms=%.1f response_ms=%.1f total_ms=%.1f",
        version_id,
        len(products),
        no_similar_product_count,
        auto_selected_count,
        len(ambiguous_items),
        preprocess_ms,
        embedding_ms,
        search_ms,
        category_search_ms,
        evidence_ms,
        decision_ms,
        llm_ms,
        response_ms,
        elapsed_ms(request_started_at),
    )
    return results


def elapsed_ms(started_at: float) -> float:
    return (perf_counter() - started_at) * 1000


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


def prediction_from_alias(item: ProductCandidates, selected: PredictionCandidate) -> PredictionItem:
    remaining_candidates = [
        candidate
        for candidate in item.candidates
        if candidate.categoryCode != selected.categoryCode
    ]
    return PredictionItem(
        rowId=item.product.rowId,
        categoryId=selected.categoryId,
        categoryCode=selected.categoryCode,
        fullPath=selected.fullPath,
        score=selected.score,
        candidates=[selected, *remaining_candidates][:CATEGORY_EMBEDDING_CANDIDATE_K],
        llmUsed=False,
        llmSelectedCategory=None,
        llmStatus="AUTO_SELECTED",
        llmStatusDetail="Configured category alias rule matched.",
        decisionSource="CATEGORY_ALIAS",
        similarProducts=item.similar_products,
        categoryDistribution=item.category_distribution,
    )


def attach_product_evidence(
    item: ProductCandidates,
    hits: list[ProductSearchHit],
    category_candidates: list[PredictionCandidate],
) -> ProductCandidates:
    evidence = build_product_evidence(hits)
    product_category_keys = {
        (category.categoryId, category.fullPath)
        for category in evidence.distribution[:PRODUCT_REPRESENTATIVE_K]
    }
    unique_category_candidates: list[PredictionCandidate] = []
    for candidate in category_candidates:
        key = (candidate.categoryId, candidate.fullPath)
        if key in product_category_keys:
            continue
        product_category_keys.add(key)
        unique_category_candidates.append(candidate)
        if len(unique_category_candidates) >= CATEGORY_EMBEDDING_CANDIDATE_K:
            break
    return ProductCandidates(
        product=item.product,
        candidates=unique_category_candidates,
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
