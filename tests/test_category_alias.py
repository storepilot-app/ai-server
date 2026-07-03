import tempfile
import unittest
from pathlib import Path

from app.category_matcher.rules.category_alias import (
    CategoryAlias,
    load_category_aliases,
    resolve_category_alias,
)


class CategoryAliasTest(unittest.TestCase):
    def setUp(self) -> None:
        self.categories = [{
            "categoryId": 2750,
            "categoryCode": "50004246",
            "fullPath": "출산/육아 > 완구/인형 > 미술놀이 > 클레이",
        }]
        self.aliases = [CategoryAlias(keyword="말랑이", category_code="50004246")]

    def test_mallangi_resolves_to_clay_category(self) -> None:
        result = resolve_category_alias("촉감놀이 말랑이 세트", self.categories, self.aliases)

        self.assertIsNotNone(result)
        self.assertEqual("50004246", result.categoryCode)
        self.assertEqual("출산/육아 > 완구/인형 > 미술놀이 > 클레이", result.fullPath)

    def test_similar_but_different_word_does_not_match(self) -> None:
        result = resolve_category_alias("그대로 말랑고구마 스틱", self.categories, self.aliases)

        self.assertIsNone(result)

    def test_alias_file_is_loaded_and_longer_keyword_has_priority(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "aliases.csv"
            path.write_text(
                "keyword,category_code,category_path\n"
                "말랑,111,기타\n"
                "말랑이,50004246,출산/육아 > 클레이\n",
                encoding="utf-8",
            )

            aliases = load_category_aliases(path)

        self.assertEqual("말랑이", aliases[0].keyword)
        self.assertEqual("50004246", aliases[0].category_code)


if __name__ == "__main__":
    unittest.main()
