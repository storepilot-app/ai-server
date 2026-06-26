import unittest

import numpy as np

from app.category_matcher.service import apply_body_keyword_category_bonus


class BodyKeywordCategoryBonusTest(unittest.TestCase):
    def test_adds_bonus_to_figure_category(self) -> None:
        scores = np.array([0.4, 0.5], dtype=np.float32)
        categories = [
            {
                "categoryId": 1,
                "categoryCode": "A",
                "fullPath": "생활/건강 > 수집품 > 모형/프라모델/피규어 > 피규어",
                "searchText": "",
            },
            {
                "categoryId": 2,
                "categoryCode": "B",
                "fullPath": "도서 > 소설",
                "searchText": "",
            },
        ]

        adjusted_scores = apply_body_keyword_category_bonus(
            "책 읽는 미피 미니피규어",
            scores,
            categories,
        )

        self.assertGreater(adjusted_scores[0], scores[0])
        self.assertLess(adjusted_scores[1], scores[1])

    def test_exact_figure_leaf_gets_extra_bonus(self) -> None:
        scores = np.array([0.4, 0.4], dtype=np.float32)
        categories = [
            {
                "categoryId": 1,
                "categoryCode": "A",
                "fullPath": "생활/건강 > 수집품 > 모형/프라모델/피규어 > 프라모델",
                "searchText": "",
            },
            {
                "categoryId": 2,
                "categoryCode": "B",
                "fullPath": "생활/건강 > 수집품 > 모형/프라모델/피규어 > 피규어",
                "searchText": "",
            },
        ]

        adjusted_scores = apply_body_keyword_category_bonus(
            "미피 미니피규어",
            scores,
            categories,
        )

        self.assertGreater(adjusted_scores[1], adjusted_scores[0])

    def test_body_keyword_has_priority_over_gunpla_like_word(self) -> None:
        scores = np.array([0.65, 0.55], dtype=np.float32)
        categories = [
            {
                "categoryId": 1,
                "categoryCode": "A",
                "fullPath": "생활/건강 > 수집품 > 모형/프라모델/피규어 > 프라모델",
                "searchText": "",
            },
            {
                "categoryId": 2,
                "categoryCode": "B",
                "fullPath": "생활/건강 > 문구/사무용품 > 필기도구 > 샤프",
                "searchText": "",
            },
        ]

        adjusted_scores = apply_body_keyword_category_bonus(
            "산리오 빙글빙글 돔 샤프",
            scores,
            categories,
        )

        self.assertGreater(adjusted_scores[1], adjusted_scores[0])


if __name__ == "__main__":
    unittest.main()
