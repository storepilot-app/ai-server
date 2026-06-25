import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from app.category_matcher.schemas import CategoryItem, PredictionCandidate, PredictionItem, ProductItem

MODEL_NAME = os.getenv("STOREPILOT_EMBEDDING_MODEL", "BAAI/bge-m3")
CACHE_ROOT = Path(os.getenv("STOREPILOT_AI_CACHE_ROOT", "ai-cache/categories"))
MODEL_CACHE_KEY = re.sub(r"[^A-Za-z0-9_.-]+", "_", MODEL_NAME).strip("_").lower()
GUNPLA_CATEGORY_BONUS = float(os.getenv("STOREPILOT_GUNPLA_CATEGORY_BONUS", "0.3"))
LLM_API_KEY = os.getenv("STOREPILOT_LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("STOREPILOT_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
LLM_MODEL = os.getenv("STOREPILOT_LLM_MODEL", "gpt-4o-mini")
LLM_TIMEOUT_SECONDS = float(os.getenv("STOREPILOT_LLM_TIMEOUT_SECONDS", "90"))
LLM_BATCH_SIZE = int(os.getenv("STOREPILOT_LLM_BATCH_SIZE", "30"))
GUNPLA_STRONG_KEYWORDS = [
    "HG",
    "MG",
    "RG",
    "PG",
    "EG",
    "SD",
    "HGUC",
    "MGEX",
    "건담",
    "건프라",
    "프라모델",
    "반다이",
    "자쿠",
    "릭돔",
    "돔",
    "사자비",
    "뉴건담",
    "유니콘",
    "스트라이크",
    "프리덤",
    "엑시아",
    "발바토스",
    "에어리얼",
    "IBO",
    "GQ",
    "SEED",
    "UC",
]

_model: SentenceTransformer | None = None
logger = logging.getLogger(__name__)


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


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def rebuild_category_cache(version_id: int, categories: list[CategoryItem]) -> None:
    version_dir = category_cache_dir(version_id)
    version_dir.mkdir(parents=True, exist_ok=True)

    passages = [category_text(category) for category in categories]
    embeddings = embed(passages)

    np.save(version_dir / "category_embeddings.npy", embeddings)
    (version_dir / "model.json").write_text(
        json.dumps({"modelName": MODEL_NAME, "dimension": int(embeddings.shape[1])}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (version_dir / "category_meta.json").write_text(
        json.dumps([category.model_dump() for category in categories], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def predict_categories(version_id: int, products: list[ProductItem]) -> list[PredictionItem]:
    embeddings, categories = load_category_cache(version_id)
    if len(categories) == 0:
        return [
            PredictionItem(rowId=product.rowId, categoryId=None, categoryCode=None, fullPath=None, score=0.0, candidates=[])
            for product in products
        ]

    queries = [preprocess_product_name(product.productName) for product in products]
    query_embeddings = embed(queries)
    if query_embeddings.shape[1] != embeddings.shape[1]:
        return [
            PredictionItem(rowId=product.rowId, categoryId=None, categoryCode=None, fullPath=None, score=0.0, candidates=[])
            for product in products
        ]

    scores = query_embeddings @ embeddings.T

    product_candidates: list[ProductCandidates] = []
    for product, row_scores in zip(products, scores):
        row_scores = apply_gunpla_category_bonus(product.productName, row_scores, categories)
        top_indexes = np.argsort(row_scores)[::-1][:10]
        candidates = [
            PredictionCandidate(
                categoryId=candidate["categoryId"],
                categoryCode=candidate["categoryCode"],
                fullPath=candidate["fullPath"],
                score=float(row_scores[int(candidate_index)]),
            )
            for candidate_index in top_indexes
            for candidate in [categories[int(candidate_index)]]
        ]
        product_candidates.append(ProductCandidates(product=product, candidates=candidates))

    llm_selections = select_candidates_with_llm_batch(product_candidates)

    results: list[PredictionItem] = []
    for item in product_candidates:
        llm_selection = llm_selections.get(
            item.product.rowId,
            fallback_llm_selection(item.candidates, "FAILED", "LLM batch response missing this rowId."),
        )
        selected_candidate = llm_selection.selected_candidate

        if selected_candidate is None:
            results.append(
                PredictionItem(
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
            )
            continue

        results.append(
            PredictionItem(
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
        )
    return results


def select_candidate_with_llm(product_name: str, candidates: list[PredictionCandidate]) -> LlmSelection:
    item = ProductCandidates(product=ProductItem(rowId=1, productName=product_name), candidates=candidates)
    return select_candidates_with_llm_batch([item]).get(1, fallback_llm_selection(candidates, "FAILED", "LLM result missing."))


def select_candidates_with_llm_batch(items: list[ProductCandidates]) -> dict[int, LlmSelection]:
    selections: dict[int, LlmSelection] = {}
    if not items:
        return selections

    for chunk in chunks(items, max(1, LLM_BATCH_SIZE)):
        selections.update(select_candidates_chunk_with_llm(chunk))
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
        item.product.rowId: selection_from_llm_decision(item.candidates, decisions_by_row_id.get(item.product.rowId))
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
                                "candidates": [
                                    {
                                        "index": index,
                                        "fullPath": candidate.fullPath,
                                        "embeddingScore": candidate.score,
                                    }
                                    for index, candidate in enumerate(item.candidates)
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
    request = urllib.request.Request(
        f"{LLM_BASE_URL}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=LLM_TIMEOUT_SECONDS) as response:
        response_body = json.loads(response.read().decode("utf-8"))

    content = response_body["choices"][0]["message"]["content"]
    parsed = parse_llm_json(content)
    results = parsed.get("results")
    if not isinstance(results, list):
        raise ValueError("LLM response did not contain results array.")
    return results


def chunks(items: list[ProductCandidates], size: int) -> list[list[ProductCandidates]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def parse_llm_json(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content or "", re.DOTALL)
        if not match:
            raise ValueError("LLM response did not contain JSON.")
        return json.loads(match.group(0))


def apply_gunpla_category_bonus(product_name: str, row_scores: np.ndarray, categories: list[dict]) -> np.ndarray:
    if not has_gunpla_keyword(product_name):
        return row_scores

    adjusted_scores = row_scores.copy()
    for index, category in enumerate(categories):
        if is_plamodel_category(category):
            adjusted_scores[index] += GUNPLA_CATEGORY_BONUS
    return adjusted_scores


def has_gunpla_keyword(product_name: str) -> bool:
    normalized = normalize_keyword_text(product_name)
    if not normalized:
        return False
    return any(normalize_keyword_text(keyword) in normalized for keyword in GUNPLA_STRONG_KEYWORDS)


def is_plamodel_category(category: dict) -> bool:
    text = f"{category.get('fullPath', '')} {category.get('searchText', '')}"
    return "프라모델" in text


def normalize_keyword_text(value: str) -> str:
    return re.sub(r"[\s_/(),\[\]{}|,-]+", "", value or "").upper()


def embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    vectors = get_model().encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def load_category_cache(version_id: int) -> tuple[np.ndarray, list[dict]]:
    version_dir = category_cache_dir(version_id)
    embeddings_path = version_dir / "category_embeddings.npy"
    meta_path = version_dir / "category_meta.json"
    if not embeddings_path.exists() or not meta_path.exists():
        return np.empty((0, 0), dtype=np.float32), []

    embeddings = np.load(embeddings_path)
    categories = json.loads(meta_path.read_text(encoding="utf-8"))
    return embeddings, categories


def category_cache_dir(version_id: int) -> Path:
    return CACHE_ROOT / MODEL_CACHE_KEY / f"version-{version_id}"


def category_text(category: CategoryItem) -> str:
    return category.searchText or category.fullPath


def preprocess_product_name(product_name: str) -> str:
    text = product_name or ""
    text = re.sub(r"\([^)]*\)|（[^）]*）", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[\[\](){}（）]", " ", text)
    text = re.sub(r"[_/|,]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
