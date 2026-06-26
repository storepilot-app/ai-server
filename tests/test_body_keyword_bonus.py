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

    def test_slime_and_mallangi_match_clay_category(self) -> None:
        scores = np.array([0.45, 0.6], dtype=np.float32)
        categories = [
            {
                "categoryId": 1,
                "categoryCode": "A",
                "fullPath": "생활/건강 > 문구/사무용품 > 미술용품 > 클레이",
                "searchText": "",
            },
            {
                "categoryId": 2,
                "categoryCode": "B",
                "fullPath": "생활/건강 > 수집품 > 피규어",
                "searchText": "",
            },
        ]

        slime_scores = apply_body_keyword_category_bonus("과일향 슬라임 세트", scores, categories)
        mallangi_scores = apply_body_keyword_category_bonus("동물 말랑이 장난감", scores, categories)

        self.assertGreater(slime_scores[0], slime_scores[1])
        self.assertGreater(mallangi_scores[0], mallangi_scores[1])

    def test_dice_matches_board_game_category(self) -> None:
        scores = np.array([0.45, 0.6], dtype=np.float32)
        categories = [
            {
                "categoryId": 1,
                "categoryCode": "A",
                "fullPath": "생활/건강 > 완구 > 보드게임 > 보드게임",
                "searchText": "",
            },
            {
                "categoryId": 2,
                "categoryCode": "B",
                "fullPath": "생활/건강 > 문구/사무용품 > 필기도구",
                "searchText": "",
            },
        ]

        adjusted_scores = apply_body_keyword_category_bonus("랜덤 주사위 세트", scores, categories)

        self.assertGreater(adjusted_scores[0], adjusted_scores[1])

    def test_bottle_opener_matches_opener_category(self) -> None:
        scores = np.array([0.45, 0.6], dtype=np.float32)
        categories = [
            {
                "categoryId": 1,
                "categoryCode": "A",
                "fullPath": "생활/건강 > 주방용품 > 조리기구 > 오프너",
                "searchText": "",
            },
            {
                "categoryId": 2,
                "categoryCode": "B",
                "fullPath": "생활/건강 > 생활용품 > 생활잡화",
                "searchText": "",
            },
        ]

        adjusted_scores = apply_body_keyword_category_bonus("휴대용 병따개 키링", scores, categories)

        self.assertGreater(adjusted_scores[0], adjusted_scores[1])

    def test_cross_matches_christian_goods_category(self) -> None:
        scores = np.array([0.45, 0.6], dtype=np.float32)
        categories = [
            {
                "categoryId": 1,
                "categoryCode": "A",
                "fullPath": "생활/건강 > 종교 > 기독교용품",
                "searchText": "",
            },
            {
                "categoryId": 2,
                "categoryCode": "B",
                "fullPath": "패션잡화 > 액세서리",
                "searchText": "",
            },
        ]

        adjusted_scores = apply_body_keyword_category_bonus("십자가 장식 소품", scores, categories)

        self.assertGreater(adjusted_scores[0], adjusted_scores[1])

    def test_pencil_cap_matches_pencil_category(self) -> None:
        scores = np.array([0.45, 0.6], dtype=np.float32)
        categories = [
            {
                "categoryId": 1,
                "categoryCode": "A",
                "fullPath": "생활/건강 > 문구/사무용품 > 필기도구 > 연필",
                "searchText": "",
            },
            {
                "categoryId": 2,
                "categoryCode": "B",
                "fullPath": "생활/건강 > 생활용품 > 생활잡화",
                "searchText": "",
            },
        ]

        adjusted_scores = apply_body_keyword_category_bonus("캐릭터 연필캡 세트", scores, categories)

        self.assertGreater(adjusted_scores[0], adjusted_scores[1])

    def test_clip_matches_clip_pin_category(self) -> None:
        scores = np.array([0.45, 0.6], dtype=np.float32)
        categories = [
            {
                "categoryId": 1,
                "categoryCode": "A",
                "fullPath": "생활/건강 > 문구/사무용품 > 문구용품 > 클립/핀",
                "searchText": "",
            },
            {
                "categoryId": 2,
                "categoryCode": "B",
                "fullPath": "생활/건강 > 생활용품 > 생활잡화",
                "searchText": "",
            },
        ]

        adjusted_scores = apply_body_keyword_category_bonus("컬러 클립 세트", scores, categories)

        self.assertGreater(adjusted_scores[0], adjusted_scores[1])


if __name__ == "__main__":
    unittest.main()
