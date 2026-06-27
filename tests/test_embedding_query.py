import unittest

from app.category_matcher.preprocess.query import preprocess_embedding_query


class EmbeddingQueryPreprocessingTest(unittest.TestCase):
    def test_keeps_natural_product_name_without_weighting(self) -> None:
        value = preprocess_embedding_query("책읽는 앉은 곰 인형")

        self.assertEqual("책읽는 앉은 곰 인형", value)

    def test_keeps_model_numbers_and_uppercase_model_units(self) -> None:
        value = preprocess_embedding_query("로지텍 K380 30MM 프라모델")

        self.assertEqual("로지텍 K380 30MM 프라모델", value)

    def test_keeps_parenthesized_meaning_but_removes_quantity(self) -> None:
        value = preprocess_embedding_query("말랑고구마(스틱)250g 3개")

        self.assertEqual("말랑고구마 스틱", value)


if __name__ == "__main__":
    unittest.main()
