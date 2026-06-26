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
        value = emphasize_noun_like_terms("책읽는 앉은 곰 인형")

        self.assertEqual("책 곰 인형", value)

    def test_removes_low_signal_adverbs(self) -> None:
        value = emphasize_noun_like_terms("너무 예쁜 키링")

        self.assertEqual("키링", value)

    def test_embedding_query_weights_later_terms_more(self) -> None:
        value = preprocess_embedding_query("책읽는 무드등")

        self.assertNotIn("책", value.split())
        self.assertGreaterEqual(value.split().count("무드등"), 4)

    def test_tail_token_weight_has_three_levels(self) -> None:
        value = apply_tail_token_weight("A B C")

        self.assertEqual("A B B C C C", value)

    def test_extracts_noun_terms_with_kiwi_when_available(self) -> None:
        def tokenize(value: str) -> list[SimpleNamespace]:
            tokens_by_value = {
                "책읽는": [
                    SimpleNamespace(form="책", tag="NNG"),
                    SimpleNamespace(form="읽", tag="VV"),
                ],
                "곰": [SimpleNamespace(form="곰", tag="NNG")],
                "인형": [SimpleNamespace(form="인형", tag="NNG")],
            }
            return tokens_by_value[value]

        kiwi = SimpleNamespace(
            tokenize=tokenize
        )

        with patch("app.category_matcher.service.get_kiwi", return_value=kiwi):
            value = extract_noun_focused_terms("책읽는 곰 인형")

        self.assertEqual("책 곰 인형", value)

    def test_detects_body_keyword(self) -> None:
        value = detect_body_keywords("책 읽는 미피 랜덤 미니피규어")

        self.assertIn("피규어", value)

    def test_body_keyword_focus_removes_context_nouns_and_boosts_body(self) -> None:
        value = apply_body_keyword_focus(
            "책 미피 랜덤 미니피규어",
            ["피규어"],
        )

        self.assertNotIn("책", value.split())
        self.assertNotIn("랜덤", value.split())
        self.assertGreaterEqual(value.split().count("피규어"), 4)


if __name__ == "__main__":
    unittest.main()
