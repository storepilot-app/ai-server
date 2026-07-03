import json
import unittest
from threading import Barrier, Lock
from unittest.mock import MagicMock, patch

from app.category_matcher.llm import judge as service
from app.category_matcher.schemas import CategoryDistributionItem, PredictionCandidate, SimilarProductItem


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

    def test_accepts_compact_llm_response(self) -> None:
        candidate = PredictionCandidate(categoryId=1, categoryCode="A", fullPath="A > B", score=0.7)

        selection = service.selection_from_llm_decision(
            [candidate],
            {"id": 1, "m": True, "i": 0},
        )

        self.assertEqual(candidate, selection.selected_candidate)
        self.assertEqual("SELECTED", selection.status)

    def test_sends_compact_payload_without_duplicate_candidate_data(self) -> None:
        item = service.ProductCandidates(
            product=service.ProductItem(rowId=7, productName="compact product"),
            candidates=[],
            similar_products=[
                SimilarProductItem(
                    productName="similar product",
                    categoryId=1,
                    categoryCode="A",
                    fullPath="A > B",
                    similarity=0.912345,
                ),
            ],
            category_distribution=[
                CategoryDistributionItem(
                    categoryId=1,
                    categoryCode="A",
                    fullPath="A > B",
                    support=0.8,
                    exampleCount=4,
                    maxSimilarity=0.912345,
                ),
            ],
        )
        response_body = {
            "choices": [{
                "message": {
                    "content": '{"results":[{"rowId":7,"matched":true,"selectedIndex":0}]}'
                }
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10},
        }
        response = MagicMock()
        response.read.return_value = json.dumps(response_body).encode("utf-8")
        context_manager = MagicMock()
        context_manager.__enter__.return_value = response

        with patch.object(service.urllib.request, "urlopen", return_value=context_manager) as urlopen:
            decisions = service.request_llm_category_decisions([item])

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        system_prompt = payload["messages"][0]["content"]
        user_payload = json.loads(payload["messages"][1]["content"])
        compact_item = user_payload["products"][0]
        self.assertIn("product name provides enough evidence", system_prompt)
        self.assertIn("Prefer a broader/general category", system_prompt)
        self.assertIn("assumptions not supported by the product name", system_prompt)
        self.assertIn("distinguish accessories from main products", system_prompt)
        self.assertIn("matched=false", system_prompt)
        self.assertEqual({"rowId", "productName", "categoryOptions"}, set(compact_item))
        self.assertNotIn("similarProducts", request.data.decode("utf-8"))
        self.assertNotIn("candidates", request.data.decode("utf-8"))
        self.assertNotIn("categoryDistribution", request.data.decode("utf-8"))
        self.assertNotIn("similar product", request.data.decode("utf-8"))
        self.assertEqual(
            [[0, "A > B", 0.9123, "SIMILAR_PRODUCT"]],
            compact_item["categoryOptions"],
        )
        self.assertEqual([{"rowId": 7, "matched": True, "selectedIndex": 0}], decisions)

    def test_uses_top_five_unique_category_options(self) -> None:
        distributions = [
            CategoryDistributionItem(
                categoryId=index,
                categoryCode=str(index),
                fullPath=f"Category > {index}",
                support=1.0 / 6,
                exampleCount=1,
                maxSimilarity=0.9 - index * 0.01,
            )
            for index in range(8)
        ]
        item = service.ProductCandidates(
            product=service.ProductItem(rowId=1, productName="product"),
            candidates=[],
            category_distribution=distributions,
        )

        options = item.selection_candidates()

        self.assertEqual(5, len(options))
        self.assertEqual([f"Category > {index}" for index in range(5)], [option.fullPath for option in options])

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
