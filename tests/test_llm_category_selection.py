import unittest
from threading import Barrier, Lock
from unittest.mock import patch

from app.category_matcher.llm import judge as service
from app.category_matcher.schemas import PredictionCandidate, SimilarProductItem


class LlmCategorySelectionTest(unittest.TestCase):
    def test_parse_json_from_llm_response(self) -> None:
        value = service.parse_llm_json('{"matched": true, "selectedIndex": 1, "confidence": 0.8, "reason": "ok"}')

        self.assertTrue(value["matched"])
        self.assertEqual(value["selectedIndex"], 1)

    def test_selects_none_when_llm_rejects_candidates(self) -> None:
        candidates = [
            PredictionCandidate(categoryId=1, categoryCode="A", fullPath="A > B", score=0.7),
        ]

        with patch.object(service, "LLM_API_KEY", "test-key"), patch.object(
            service,
            "request_llm_category_decisions",
            return_value=[{"rowId": 1, "matched": False, "selectedIndex": None, "confidence": 0.2, "reason": "no match"}],
        ):
            selection = service.select_candidate_with_llm("unrelated product", candidates)

        self.assertIsNone(selection.selected_candidate)
        self.assertTrue(selection.used)
        self.assertEqual(selection.status, "REJECTED")

    def test_selects_llm_candidate_index(self) -> None:
        candidates = [
            PredictionCandidate(categoryId=1, categoryCode="A", fullPath="A > B", score=0.7),
            PredictionCandidate(categoryId=2, categoryCode="C", fullPath="C > D", score=0.6),
        ]

        with patch.object(service, "LLM_API_KEY", "test-key"), patch.object(
            service,
            "request_llm_category_decisions",
            return_value=[{"rowId": 1, "matched": True, "selectedIndex": 1, "confidence": 0.9, "reason": "best"}],
        ):
            selection = service.select_candidate_with_llm("product", candidates)

        self.assertEqual(selection.selected_candidate, candidates[1])
        self.assertTrue(selection.used)
        self.assertEqual(selection.status, "SELECTED")

    def test_batches_multiple_products_in_one_llm_request(self) -> None:
        items = [
            service.ProductCandidates(
                product=service.ProductItem(rowId=10, productName="first"),
                candidates=[PredictionCandidate(categoryId=1, categoryCode="A", fullPath="A > B", score=0.7)],
            ),
            service.ProductCandidates(
                product=service.ProductItem(rowId=20, productName="second"),
                candidates=[PredictionCandidate(categoryId=2, categoryCode="C", fullPath="C > D", score=0.6)],
            ),
        ]

        with patch.object(service, "LLM_API_KEY", "test-key"), patch.object(
            service,
            "LLM_BATCH_SIZE",
            30,
        ), patch.object(
            service,
            "request_llm_category_decisions",
            return_value=[
                {"rowId": 10, "matched": True, "selectedIndex": 0, "confidence": 0.9, "reason": "best"},
                {"rowId": 20, "matched": False, "selectedIndex": None, "confidence": 0.2, "reason": "no match"},
            ],
        ) as request:
            selections = service.select_candidates_with_llm_batch(items)

        request.assert_called_once()
        self.assertEqual(selections[10].status, "SELECTED")
        self.assertEqual(selections[20].status, "REJECTED")

    def test_runs_llm_chunks_concurrently_up_to_configured_limit(self) -> None:
        items = [
            service.ProductCandidates(
                product=service.ProductItem(rowId=row_id, productName=f"product-{row_id}"),
                candidates=[PredictionCandidate(categoryId=row_id, categoryCode=str(row_id), fullPath="A > B", score=0.7)],
            )
            for row_id in range(1, 5)
        ]
        barrier = Barrier(4)
        lock = Lock()
        active_requests = 0
        maximum_active_requests = 0

        def concurrent_response(chunk: list[service.ProductCandidates]) -> list[dict]:
            nonlocal active_requests, maximum_active_requests
            with lock:
                active_requests += 1
                maximum_active_requests = max(maximum_active_requests, active_requests)
            try:
                barrier.wait(timeout=2)
                return [{
                    "rowId": chunk[0].product.rowId,
                    "matched": True,
                    "selectedIndex": 0,
                    "confidence": 0.9,
                    "reason": "best",
                }]
            finally:
                with lock:
                    active_requests -= 1

        with patch.object(service, "LLM_API_KEY", "test-key"), patch.object(
            service,
            "LLM_BATCH_SIZE",
            1,
        ), patch.object(
            service,
            "LLM_MAX_CONCURRENCY",
            4,
        ), patch.object(
            service,
            "request_llm_category_decisions",
            side_effect=concurrent_response,
        ):
            selections = service.select_candidates_with_llm_batch(items)

        self.assertEqual(4, maximum_active_requests)
        self.assertEqual({1, 2, 3, 4}, set(selections))

    def test_selects_category_from_similar_product_candidates(self) -> None:
        item = service.ProductCandidates(
            product=service.ProductItem(rowId=30, productName="말랑고구마 스틱"),
            candidates=[
                PredictionCandidate(categoryId=1, categoryCode="A", fullPath="식품 > 고구마", score=0.7),
            ],
            similar_products=[
                SimilarProductItem(
                    productName="강아지 고구마 스틱",
                    categoryId=2,
                    categoryCode="B",
                    fullPath="반려동물 > 강아지 간식 > 트릿/스틱",
                    similarity=0.89,
                ),
            ],
        )

        with patch.object(service, "LLM_API_KEY", "test-key"), patch.object(
            service,
            "request_llm_category_decisions",
            return_value=[{"rowId": 30, "matched": True, "selectedIndex": 0, "confidence": 0.9, "reason": "best"}],
        ):
            selection = service.select_candidates_with_llm_batch([item])[30]

        self.assertEqual(2, selection.selected_candidate.categoryId)
        self.assertEqual("반려동물 > 강아지 간식 > 트릿/스틱", selection.selected_candidate.fullPath)


if __name__ == "__main__":
    unittest.main()
