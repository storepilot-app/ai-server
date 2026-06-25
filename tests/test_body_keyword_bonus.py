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
                "fullPath": "\uc0dd\ud65c/\uac74\uac15 > \uc218\uc9d1\ud488 > \ubaa8\ud615/\ud504\ub77c\ubaa8\ub378/\ud53c\uaddc\uc5b4 > \ud53c\uaddc\uc5b4",
                "searchText": "",
            },
            {
                "categoryId": 2,
                "categoryCode": "B",
                "fullPath": "\ub3c4\uc11c > \uc18c\uc124",
                "searchText": "",
            },
        ]

        adjusted_scores = apply_body_keyword_category_bonus(
            "\ucc45 \uc77d\ub294 \ubbf8\ud53c \ubbf8\ub2c8\ud53c\uaddc\uc5b4",
            scores,
            categories,
        )

        self.assertGreater(adjusted_scores[0], scores[0])
        self.assertEqual(float(adjusted_scores[1]), float(scores[1]))

    def test_exact_figure_leaf_gets_extra_bonus(self) -> None:
        scores = np.array([0.4, 0.4], dtype=np.float32)
        categories = [
            {
                "categoryId": 1,
                "categoryCode": "A",
                "fullPath": "\uc0dd\ud65c/\uac74\uac15 > \uc218\uc9d1\ud488 > \ubaa8\ud615/\ud504\ub77c\ubaa8\ub378/\ud53c\uaddc\uc5b4 > \ud504\ub77c\ubaa8\ub378",
                "searchText": "",
            },
            {
                "categoryId": 2,
                "categoryCode": "B",
                "fullPath": "\uc0dd\ud65c/\uac74\uac15 > \uc218\uc9d1\ud488 > \ubaa8\ud615/\ud504\ub77c\ubaa8\ub378/\ud53c\uaddc\uc5b4 > \ud53c\uaddc\uc5b4",
                "searchText": "",
            },
        ]

        adjusted_scores = apply_body_keyword_category_bonus(
            "\ubbf8\ud53c \ubbf8\ub2c8\ud53c\uaddc\uc5b4",
            scores,
            categories,
        )

        self.assertGreater(adjusted_scores[1], adjusted_scores[0])


if __name__ == "__main__":
    unittest.main()
