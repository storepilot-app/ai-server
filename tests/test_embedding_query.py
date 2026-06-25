import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.category_matcher.service import (
    apply_body_keyword_focus,
    apply_tail_token_weight,
    detect_body_keywords,
    emphasize_noun_like_terms,
    extract_noun_focused_terms,
    preprocess_embedding_query,
)


class EmbeddingQueryPreprocessingTest(unittest.TestCase):
    def test_downweights_verb_like_modifiers(self) -> None:
        value = emphasize_noun_like_terms("\ucc45\uc77d\ub294 \uc549\uc740 \uacf0 \uc778\ud615")

        self.assertEqual("\ucc45 \uacf0 \uc778\ud615", value)

    def test_removes_low_signal_adverbs(self) -> None:
        value = emphasize_noun_like_terms("\ub108\ubb34 \uc608\uc05c \ud0a4\ub9c1")

        self.assertEqual("\ud0a4\ub9c1", value)

    def test_embedding_query_weights_later_terms_more(self) -> None:
        value = preprocess_embedding_query("\ucc45\uc77d\ub294 \ubb34\ub4dc\ub4f1")

        self.assertNotIn("\ucc45", value.split())
        self.assertGreaterEqual(value.split().count("\ubb34\ub4dc\ub4f1"), 4)

    def test_tail_token_weight_has_three_levels(self) -> None:
        value = apply_tail_token_weight("A B C")

        self.assertEqual("A B B C C C", value)

    def test_extracts_noun_terms_with_kiwi_when_available(self) -> None:
        def tokenize(value: str) -> list[SimpleNamespace]:
            tokens_by_value = {
                "\ucc45\uc77d\ub294": [
                    SimpleNamespace(form="\ucc45", tag="NNG"),
                    SimpleNamespace(form="\uc77d", tag="VV"),
                ],
                "\uacf0": [SimpleNamespace(form="\uacf0", tag="NNG")],
                "\uc778\ud615": [SimpleNamespace(form="\uc778\ud615", tag="NNG")],
            }
            return tokens_by_value[value]

        kiwi = SimpleNamespace(
            tokenize=tokenize
        )

        with patch("app.category_matcher.service.get_kiwi", return_value=kiwi):
            value = extract_noun_focused_terms("\ucc45\uc77d\ub294 \uacf0 \uc778\ud615")

        self.assertEqual("\ucc45 \uacf0 \uc778\ud615", value)

    def test_detects_body_keyword(self) -> None:
        value = detect_body_keywords("\ucc45 \uc77d\ub294 \ubbf8\ud53c \ub79c\ub364 \ubbf8\ub2c8\ud53c\uaddc\uc5b4")

        self.assertIn("\ud53c\uaddc\uc5b4", value)

    def test_body_keyword_focus_removes_context_nouns_and_boosts_body(self) -> None:
        value = apply_body_keyword_focus(
            "\ucc45 \ubbf8\ud53c \ub79c\ub364 \ubbf8\ub2c8\ud53c\uaddc\uc5b4",
            ["\ud53c\uaddc\uc5b4"],
        )

        self.assertNotIn("\ucc45", value.split())
        self.assertNotIn("\ub79c\ub364", value.split())
        self.assertGreaterEqual(value.split().count("\ud53c\uaddc\uc5b4"), 4)


if __name__ == "__main__":
    unittest.main()
