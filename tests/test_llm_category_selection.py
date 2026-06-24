import unittest
from unittest.mock import patch

from app.category_matcher import service
from app.category_matcher.schemas import PredictionCandidate


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


if __name__ == "__main__":
    unittest.main()
