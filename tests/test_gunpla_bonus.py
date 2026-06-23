import unittest

import numpy as np

from app.category_matcher.service import apply_gunpla_category_bonus, has_gunpla_keyword


class GunplaCategoryBonusTest(unittest.TestCase):
    def test_detects_gunpla_keyword(self) -> None:
        self.assertTrue(has_gunpla_keyword("HG RX-78 \uac74\ub2f4"))

    def test_adds_bonus_only_to_plamodel_category(self) -> None:
        scores = np.array([0.4, 0.5], dtype=np.float32)
        categories = [
            {
                "categoryId": 1,
                "categoryCode": "A",
                "fullPath": "\ucde8\ubbf8 > \ud504\ub77c\ubaa8\ub378",
                "searchText": "",
            },
            {
                "categoryId": 2,
                "categoryCode": "B",
                "fullPath": "\uc0dd\ud65c > \uc7a1\ud654",
                "searchText": "",
            },
        ]

        adjusted_scores = apply_gunpla_category_bonus("MG \uc0ac\uc790\ube44", scores, categories)

        self.assertGreater(adjusted_scores[0], scores[0])
        self.assertEqual(float(adjusted_scores[1]), float(scores[1]))
        self.assertAlmostEqual(float(scores[0]), 0.4)


if __name__ == "__main__":
    unittest.main()
