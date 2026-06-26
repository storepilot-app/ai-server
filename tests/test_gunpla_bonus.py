import unittest

import numpy as np

from app.category_matcher.service import apply_gunpla_category_bonus, has_gunpla_keyword


class GunplaCategoryBonusTest(unittest.TestCase):
    def test_detects_gunpla_keyword(self) -> None:
        self.assertTrue(has_gunpla_keyword("HG RX-78 건담"))
        self.assertTrue(has_gunpla_keyword("30MM 옵션 파츠 세트"))

    def test_adds_bonus_only_to_plamodel_category(self) -> None:
        scores = np.array([0.4, 0.5], dtype=np.float32)
        categories = [
            {
                "categoryId": 1,
                "categoryCode": "A",
                "fullPath": "취미 > 프라모델",
                "searchText": "",
            },
            {
                "categoryId": 2,
                "categoryCode": "B",
                "fullPath": "생활 > 잡화",
                "searchText": "",
            },
        ]

        adjusted_scores = apply_gunpla_category_bonus("MG 사자비", scores, categories)

        self.assertGreater(adjusted_scores[0], scores[0])
        self.assertEqual(float(adjusted_scores[1]), float(scores[1]))
        self.assertAlmostEqual(float(scores[0]), 0.4)


if __name__ == "__main__":
    unittest.main()
