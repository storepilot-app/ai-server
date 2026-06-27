import unittest

from app.category_matcher.service import preprocess_product_name


class ProductNamePreprocessingTest(unittest.TestCase):
    def test_keeps_parenthesized_text_and_model_numbers(self) -> None:
        value = preprocess_product_name("로지텍 K380 키보드 (정품 2개 세트)")

        self.assertEqual("로지텍 K380 키보드 정품 세트", value)

    def test_normalizes_full_width_parentheses_without_removing_content(self) -> None:
        value = preprocess_product_name("무선 키보드（해외배송 2026）")

        self.assertEqual("무선 키보드 해외배송 2026", value)


if __name__ == "__main__":
    unittest.main()
