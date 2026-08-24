import unittest
from unittest.mock import patch

import numpy as np

from app.category_matcher import service
from app.category_matcher.decision.policy import auto_accepted_category
from app.category_matcher.llm.judge import ProductCandidates
from app.category_matcher.product_memory.store import NaverCategoryLabel, ProductSearchHit
from app.category_matcher.retrieval.evidence import build_product_evidence
from app.category_matcher.schemas import PredictionCandidate, ProductItem


class ProductEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.category_a = NaverCategoryLabel(1, "NAVER-A", "생활 > 문구 > 계산기")
        self.category_b = NaverCategoryLabel(2, "NAVER-B", "생활 > 완구 > 보드게임")

    def test_distribution_uses_all_hits_and_options_have_one_product_per_category(self):
        hits = [
            ProductSearchHit(f"계산기 {index}", (self.category_a,), 0.99 - index * 0.001)
            for index in range(5)
        ] + [ProductSearchHit("주사위", (self.category_b,), 0.90)]

        evidence = build_product_evidence(hits)

        self.assertEqual(2, len(evidence.similar_products))
        self.assertEqual(1, sum(item.categoryId == 1 for item in evidence.similar_products))
        self.assertEqual(2, len(evidence.distribution))
        self.assertEqual(5, evidence.distribution[0].exampleCount)
        self.assertEqual(
            [category.categoryId for category in evidence.distribution[:5]],
            [product.categoryId for product in evidence.similar_products],
        )

    def test_conflicting_title_is_excluded_from_category_evidence(self):
        hits = [ProductSearchHit("충돌 상품", (self.category_a, self.category_b), 0.99)]

        evidence = build_product_evidence(hits)

        self.assertEqual([], evidence.similar_products)
        self.assertEqual([], evidence.distribution)

    def test_auto_accept_requires_similarity_consensus_and_examples(self):
        hits = [
            ProductSearchHit(f"전자 계산기 {index}", (self.category_a,), 0.99 - index * 0.005)
            for index in range(3)
        ]
        evidence = build_product_evidence(hits)

        accepted = auto_accepted_category(evidence.distribution)

        self.assertIsNotNone(accepted)
        self.assertEqual(1, accepted.categoryId)

    def test_category_embedding_candidates_exclude_product_categories_and_fill_five(self):
        item = ProductCandidates(product=ProductItem(rowId=1, productName="product"), candidates=[])
        hits = [ProductSearchHit("product A", (self.category_a,), 0.95)]
        direct_candidates = [
            PredictionCandidate(categoryId=1, categoryCode="NAVER-A", fullPath=self.category_a.full_path, score=0.99),
        ] + [
            PredictionCandidate(
                categoryId=index,
                categoryCode=f"NAVER-{index}",
                fullPath=f"Category > {index}",
                score=0.90 - index * 0.01,
            )
            for index in range(3, 9)
        ]

        attached = service.attach_product_evidence(item, hits, direct_candidates)

        self.assertEqual(5, len(attached.candidates))
        self.assertNotIn(1, [candidate.categoryId for candidate in attached.candidates])
        self.assertEqual([3, 4, 5, 6, 7], [candidate.categoryId for candidate in attached.candidates])

    def test_predict_skips_llm_when_product_evidence_is_auto_accepted(self):
        hits = [[
            ProductSearchHit(f"전자 계산기 {index}", (self.category_a,), 0.99 - index * 0.005)
            for index in range(3)
        ]]
        with patch.object(
            service, "embed_queries", return_value=np.asarray([[1.0, 0.0]], dtype=np.float32)
        ), patch.object(service, "search_similar_products_by_vectors", return_value=hits), patch.object(
            service, "select_candidates_with_llm_batch", return_value={}
        ) as llm:
            result = service.predict_categories(
                1,
                [ProductItem(rowId=1, productName="휴대용 전자 계산기")],
            )

        self.assertEqual("PRODUCT_AUTO_ACCEPT", result[0].decisionSource)
        self.assertEqual("AUTO_SELECTED", result[0].llmStatus)
        self.assertEqual([], result[0].candidates)
        llm.assert_called_once_with([])

    def test_predict_returns_no_similar_products_without_category_search_or_llm(self):
        with patch.object(
            service, "embed_queries", return_value=np.asarray([[1.0, 0.0]], dtype=np.float32)
        ), patch.object(service, "search_similar_products_by_vectors", return_value=[[]]), patch.object(
            service, "select_candidates_with_llm_batch", return_value={}
        ) as llm:
            result = service.predict_categories(
                1,
                [ProductItem(rowId=1, productName="검색 결과 없는 상품")],
            )

        self.assertEqual("NO_SIMILAR_PRODUCTS", result[0].llmStatus)
        self.assertEqual("NO_SIMILAR_PRODUCTS", result[0].decisionSource)
        self.assertIsNone(result[0].categoryId)
        llm.assert_called_once_with([])

    def test_mallangi_alias_auto_selects_clay_without_llm(self):
        categories = [{
            "categoryId": 2750,
            "categoryCode": "50004246",
            "fullPath": "출산/육아 > 완구/인형 > 미술놀이 > 클레이",
        }]
        with patch.object(
            service, "embed_queries", return_value=np.asarray([[1.0, 0.0]], dtype=np.float32)
        ), patch.object(
            service, "search_similar_products_by_vectors", return_value=[[]]
        ), patch.object(
            service, "search_category_candidates_by_vectors", return_value=[[]]
        ), patch.object(
            service, "load_category_metadata", return_value=categories
        ), patch.object(
            service, "select_candidates_with_llm_batch", return_value={}
        ) as llm:
            result = service.predict_categories(
                1,
                [ProductItem(rowId=1, productName="촉감놀이 말랑이 세트")],
            )

        self.assertEqual("CATEGORY_ALIAS", result[0].decisionSource)
        self.assertEqual("AUTO_SELECTED", result[0].llmStatus)
        self.assertEqual("50004246", result[0].categoryCode)
        llm.assert_called_once_with([])


if __name__ == "__main__":
    unittest.main()
