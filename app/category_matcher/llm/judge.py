import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from time import perf_counter

from app.category_matcher.schemas import (
    CategoryDistributionItem,
    PredictionCandidate,
    ProductItem,
    SimilarProductItem,
)
from app.category_matcher.config.settings import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_BATCH_SIZE,
    LLM_MODEL,
    LLM_THRESHOLD,
    LLM_TIMEOUT_SECONDS,
)


logger = logging.getLogger("uvicorn.error").getChild("storepilot.category_matcher.llm")


@dataclass(frozen=True)
class LlmSelection:
    selected_candidate: PredictionCandidate | None
    used: bool
    status: str
    detail: str | None = None


@dataclass(frozen=True)
class ProductCandidates:
    product: ProductItem
    candidates: list[PredictionCandidate]
    similar_products: list[SimilarProductItem] = field(default_factory=list)
    category_distribution: list[CategoryDistributionItem] = field(default_factory=list)

    def selection_candidates(self) -> list[PredictionCandidate]:
        if not self.similar_products:
            return self.candidates
        return [
            PredictionCandidate(
                categoryId=product.categoryId,
                categoryCode=product.categoryCode,
                fullPath=product.fullPath,
                score=product.similarity,
            )
            for product in self.similar_products
        ]


def select_candidate_with_llm(product_name: str, candidates: list[PredictionCandidate]) -> LlmSelection:
    item = ProductCandidates(product=ProductItem(rowId=1, productName=product_name), candidates=candidates)
    return select_candidates_with_llm_batch([item]).get(1, fallback_llm_selection(candidates, "FAILED", "LLM result missing."))


def select_candidates_with_llm_batch(items: list[ProductCandidates]) -> dict[int, LlmSelection]:
    selections: dict[int, LlmSelection] = {}
    if not items:
        return selections

    batch_started_at = perf_counter()
    item_chunks = list(chunks(items, max(1, LLM_BATCH_SIZE)))
    for chunk_index, chunk in enumerate(item_chunks, start=1):
        chunk_started_at = perf_counter()
        chunk_selections = select_candidates_chunk_with_llm(chunk)
        selections.update(chunk_selections)
        logger.info(
            "llm_category_chunk_timing chunk=%d/%d items=%d selected=%d rejected=%d failed=%d elapsed_ms=%.1f",
            chunk_index,
            len(item_chunks),
            len(chunk),
            sum(selection.status == "SELECTED" for selection in chunk_selections.values()),
            sum(selection.status == "REJECTED" for selection in chunk_selections.values()),
            sum(selection.status == "FAILED" for selection in chunk_selections.values()),
            elapsed_ms(chunk_started_at),
        )
    logger.info(
        "llm_category_batch_timing items=%d chunks=%d elapsed_ms=%.1f",
        len(items),
        len(item_chunks),
        elapsed_ms(batch_started_at),
    )
    return selections


def select_candidates_chunk_with_llm(items: list[ProductCandidates]) -> dict[int, LlmSelection]:
    if not LLM_API_KEY:
        return {
            item.product.rowId: fallback_llm_selection(item.candidates, "SKIPPED", "LLM API key is not set.")
            for item in items
        }

    try:
        decisions = request_llm_category_decisions(items)
    except Exception as error:
        detail = summarize_llm_error(error)
        logger.warning("LLM category judge batch failed: %s", detail)
        return {
            item.product.rowId: fallback_llm_selection(item.candidates, "FAILED", detail)
            for item in items
        }

    decisions_by_row_id = {
        decision.get("rowId"): decision
        for decision in decisions
        if isinstance(decision, dict)
    }
    return {
        item.product.rowId: selection_from_llm_decision(
            item.selection_candidates(),
            decisions_by_row_id.get(item.product.rowId),
        )
        for item in items
    }


def fallback_llm_selection(candidates: list[PredictionCandidate], status: str, detail: str | None = None) -> LlmSelection:
    if not candidates:
        return LlmSelection(selected_candidate=None, used=False, status=status, detail=detail)
    return LlmSelection(selected_candidate=candidates[0], used=False, status=status, detail=detail)


def selection_from_llm_decision(candidates: list[PredictionCandidate], decision: dict | None) -> LlmSelection:
    if decision is None:
        return fallback_llm_selection(candidates, "FAILED", "LLM response did not include this rowId.")
    if decision.get("matched") is not True:
        return LlmSelection(selected_candidate=None, used=True, status="REJECTED")

    selected_index = decision.get("selectedIndex")
    if not isinstance(selected_index, int) or selected_index < 0 or selected_index >= len(candidates):
        return LlmSelection(
            selected_candidate=candidates[0],
            used=False,
            status="FAILED",
            detail=f"Invalid selectedIndex from LLM: {selected_index}",
        )

    return LlmSelection(selected_candidate=candidates[selected_index], used=True, status="SELECTED")


def summarize_llm_error(error: Exception) -> str:
    if isinstance(error, urllib.error.HTTPError):
        body = error.read().decode("utf-8", errors="replace")
        message = extract_openai_error_message(body)
        return f"HTTP {error.code}: {message}" if message else f"HTTP {error.code}: {error.reason}"
    if isinstance(error, urllib.error.URLError):
        return f"URL error: {error.reason}"
    return f"{type(error).__name__}: {error}"


def extract_openai_error_message(body: str) -> str:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body[:200]

    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        code = error.get("code")
        if message and code:
            return f"{message} ({code})"
        if message:
            return str(message)
    return body[:200]


def request_llm_category_decisions(items: list[ProductCandidates]) -> list[dict]:
    payload = {
        "model": LLM_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict Naver shopping category judge. "
                    "Choose the best category only from the similar-product candidates. "
                    "Use the category distribution as supporting evidence. "
                    "A high similarity alone is not proof when nearby products disagree. "
                    "For each item, prefer choosing the single best category when one candidate is clearly better than the others. "
                    "Reject all candidates only when every candidate is unrelated or too broad. "
                    "Return compact JSON only with key results. results must be an array of objects with keys: "
                    "rowId(integer), matched(boolean), selectedIndex(integer or null), confidence(number from 0 to 1), reason(string). "
                    "Keep each reason under 20 Korean characters."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "items": [
                            {
                                "rowId": item.product.rowId,
                                "productName": item.product.productName,
                                "similarProductEvidenceStrong": bool(
                                    item.similar_products
                                    and item.similar_products[0].similarity >= LLM_THRESHOLD
                                ),
                                "similarProducts": [
                                    {
                                        "productName": product.productName,
                                        "category": product.fullPath,
                                        "similarity": round(product.similarity, 6),
                                    }
                                    for product in item.similar_products
                                ],
                                "categoryDistribution": [
                                    {
                                        "category": distribution.fullPath,
                                        "support": round(distribution.support, 6),
                                        "exampleCount": distribution.exampleCount,
                                        "maxSimilarity": round(distribution.maxSimilarity, 6),
                                    }
                                    for distribution in item.category_distribution
                                ],
                                "selectionSource": "SIMILAR_PRODUCTS",
                                "candidates": [
                                    _llm_candidate_payload(item, index)
                                    for index, _ in enumerate(item.selection_candidates())
                                ],
                            }
                            for item in items
                        ],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    request_data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{LLM_BASE_URL}/chat/completions",
        data=request_data,
        headers={
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    request_started_at = perf_counter()
    with urllib.request.urlopen(request, timeout=LLM_TIMEOUT_SECONDS) as response:
        response_bytes = response.read()
        response_body = json.loads(response_bytes.decode("utf-8"))
    logger.info(
        "openai_category_request_timing model=%s items=%d request_bytes=%d response_bytes=%d elapsed_ms=%.1f",
        LLM_MODEL,
        len(items),
        len(request_data),
        len(response_bytes),
        elapsed_ms(request_started_at),
    )

    content = response_body["choices"][0]["message"]["content"]
    parsed = parse_llm_json(content)
    results = parsed.get("results")
    if not isinstance(results, list):
        raise ValueError("LLM response did not contain results array.")
    return results


def elapsed_ms(started_at: float) -> float:
    return (perf_counter() - started_at) * 1000


def _llm_candidate_payload(item: ProductCandidates, index: int) -> dict:
    candidate = item.selection_candidates()[index]
    payload = {
        "index": index,
        "fullPath": candidate.fullPath,
    }
    if item.similar_products:
        product = item.similar_products[index]
        payload["similarProductName"] = product.productName
        payload["similarity"] = product.similarity
    else:
        payload["embeddingScore"] = candidate.score
    return payload


def chunks(items: list[ProductCandidates], size: int) -> list[list[ProductCandidates]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def parse_llm_json(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        import re

        match = re.search(r"\{.*\}", content or "", re.DOTALL)
        if not match:
            raise ValueError("LLM response did not contain JSON.")
        return json.loads(match.group(0))
