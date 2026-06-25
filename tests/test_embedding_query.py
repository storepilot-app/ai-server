import unittest

from app.category_matcher.service import apply_tail_token_weight, emphasize_noun_like_terms, preprocess_embedding_query


class EmbeddingQueryPreprocessingTest(unittest.TestCase):
    def test_downweights_verb_like_modifiers(self) -> None:
        value = emphasize_noun_like_terms("\ucc45\uc77d\ub294 \uc549\uc740 \uacf0 \uc778\ud615")

        self.assertEqual("\ucc45 \uacf0 \uc778\ud615", value)

    def test_removes_low_signal_adverbs(self) -> None:
        value = emphasize_noun_like_terms("\ub108\ubb34 \uc608\uc05c \ud0a4\ub9c1")

        self.assertEqual("\ud0a4\ub9c1", value)

    def test_embedding_query_weights_later_terms_more(self) -> None:
        value = preprocess_embedding_query("\ucc45\uc77d\ub294 \ubb34\ub4dc\ub4f1")

        self.assertEqual("\ucc45 \ubb34\ub4dc\ub4f1 \ubb34\ub4dc\ub4f1 \ubb34\ub4dc\ub4f1", value)

    def test_tail_token_weight_has_three_levels(self) -> None:
        value = apply_tail_token_weight("A B C")

        self.assertEqual("A B B C C C", value)


if __name__ == "__main__":
    unittest.main()
