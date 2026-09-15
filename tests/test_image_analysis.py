import json
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from app.category_matcher import service
from app.category_matcher.llm import image_analyzer as analyzer
from app.category_matcher.llm.judge import ProductCandidates, LlmSelection
from app.category_matcher.schemas import ImageProductAnalysis, ProductItem, PredictionCandidate


class ImageAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.product = ProductItem(rowId=7, productName="캐릭터밴드", imageUrl="https://example.com/image.jpg")
        self.analysis = ImageProductAnalysis(productType="반창고", colors=["파란색"],
                                            imageMatchesProductName=True, confidence=0.9)

    def test_disabled_and_missing_url_do_not_call_provider(self):
        item = ProductCandidates(self.product, [])
        with patch.object(analyzer, "ENABLED", False), patch.object(analyzer, "request_analysis") as request:
            analyzer.analyze_ambiguous_products([item])
            request.assert_not_called()
        item.product = ProductItem(rowId=7, productName="밴드")
        with patch.object(analyzer, "ENABLED", True), patch.object(analyzer, "API_KEY", "test"), \
                patch.object(analyzer, "request_analysis") as request:
            analyzer.analyze_ambiguous_products([item])
            request.assert_not_called()
        self.assertEqual("NO_IMAGE", item.image_analysis_status)

    def test_failure_and_mismatch_leave_text_only_evidence(self):
        for outcome in (TimeoutError(), self.analysis.model_copy(update={"imageMatchesProductName": False})):
            item = ProductCandidates(self.product, [])
            with patch.object(analyzer, "ENABLED", True), patch.object(analyzer, "API_KEY", "test"), \
                    patch.object(analyzer, "request_analysis") as request:
                if isinstance(outcome, Exception):
                    request.side_effect = outcome
                else:
                    request.return_value = outcome
                analyzer.analyze_ambiguous_products([item])
            self.assertIsNone(item.image_analysis)
            self.assertIn(item.image_analysis_status, {"FAILED", "UNSUITABLE"})

    def test_provider_request_contains_image_and_validates_response(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "choices": [{"message": {"content": self.analysis.model_dump_json()}}]
        }).encode()
        with patch.object(analyzer.urllib.request, "urlopen", return_value=response) as open_request:
            result = analyzer.request_analysis(self.product, 1)
        request = open_request.call_args.args[0]
        self.assertEqual("https://api.openai.com/v1/chat/completions", request.full_url)
        content = json.loads(request.data)["messages"][1]["content"]
        self.assertEqual(self.product.imageUrl, content[1]["image_url"]["url"])
        self.assertEqual(self.analysis, result)

    def test_rejects_nonpublic_and_invalid_urls(self):
        for url in ("file:///etc/passwd", "http://127.0.0.1/a", "http://169.254.169.254/a",
                    "http://localhost/a", "https://u:p@example.com/a", "http://example.com:8080/a"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                analyzer.validate_image_url(url)

    def test_only_ambiguous_products_receive_analysis_and_share_it_with_result(self):
        candidate = PredictionCandidate(categoryId=1, categoryCode="A", fullPath="의료 > 반창고", score=0.8)
        for alias in (None, candidate):
            def analyze(items):
                for item in items:
                    item.image_analysis = self.analysis
                    item.image_analysis_status = "USED"

            def judge(items):
                if items:
                    self.assertEqual(self.analysis, items[0].image_analysis)
                return {7: LlmSelection(candidate, True, "SELECTED")}

            with patch.object(service, "embed_passages", return_value=np.array([[1., 0.]])), \
                    patch.object(service, "uses_asymmetric_embeddings", return_value=False), \
                    patch.object(service, "search_similar_products_by_vectors", return_value=[[]]), \
                    patch.object(service, "search_category_candidates_by_vectors", return_value=[[candidate]]), \
                    patch.object(service, "load_category_metadata", return_value=[]), \
                    patch.object(service, "load_category_aliases", return_value=[]), \
                    patch.object(service, "resolve_category_alias", return_value=alias), \
                    patch.object(service, "analyze_ambiguous_products", side_effect=analyze) as image_call, \
                    patch.object(service, "select_candidates_with_llm_batch", side_effect=judge):
                result = service.predict_categories(1, [self.product])[0]
            self.assertEqual(0 if alias else 1, len(image_call.call_args.args[0]))
            self.assertEqual(None if alias else self.analysis, result.imageAnalysis)
