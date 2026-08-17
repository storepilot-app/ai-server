import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.category_matcher.product_memory import store


class ProductIndexStoreTest(unittest.TestCase):
    def setUp(self):
        store._loaded_index = None

    def tearDown(self):
        store._loaded_index = None

    def test_rebuild_search_and_feedback(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            workbook = root / "products.xlsx"
            self._write_workbook(
                workbook,
                [
                    ("전자 계산기", "MY-A"),
                    ("전자 계산기", "MY-A"),
                    ("보드게임 주사위", "MY-B"),
                ],
                product_column="B",
                category_column="F",
            )

            with patch.object(store, "PRODUCT_CACHE_ROOT", root / "cache"), patch.object(
                store, "embed", side_effect=self._fake_embed
            ):
                category_a = store.NaverCategoryLabel(1, "NAVER-A", "생활 > 문구 > 계산기")
                category_b = store.NaverCategoryLabel(2, "NAVER-B", "생활 > 완구 > 보드게임")
                category_c = store.NaverCategoryLabel(3, "NAVER-C", "생활 > 문구 > 전자계산기")
                result = store.rebuild_product_index(
                    [workbook],
                    {"MY-A": category_a, "MY-B": category_b},
                )
                provider_cache_key = store.get_embedding_provider().cache_key
                metadata_path = root / "cache" / provider_cache_key / "shared" / "products.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                hits = store.search_similar_products("휴대용 계산기")
                count = store.add_product_feedback("전자 계산기", category_c)
                batch_count = store.add_product_feedbacks([
                    ("새 계산기", category_c),
                    ("보드게임 주사위", category_b),
                ])
                loaded = store._load_index()

            self.assertEqual(3, result.valid_row_count)
            self.assertEqual(2, result.indexed_product_count)
            self.assertEqual(1, result.duplicate_row_count)
            self.assertEqual(2, metadata["schemaVersion"])
            self.assertIn("naverCategories", metadata["products"][0])
            self.assertNotIn("myCategoryCodes", metadata["products"][0])
            self.assertEqual("전자 계산기", hits[0].product_name)
            self.assertEqual(2, count)
            self.assertEqual(3, batch_count)
            self.assertEqual((category_c,), loaded.products[0].categories)

    @staticmethod
    def _fake_embed(texts):
        vectors = []
        for text in texts:
            if "계산기" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "주사위" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)

    @staticmethod
    def _write_workbook(
        path: Path,
        rows: list[tuple[str, str]],
        product_column: str = "D",
        category_column: str = "T",
    ) -> None:
        xml_rows = [
            f'<row r="1"><c r="{product_column}1" t="inlineStr"><is><t>상품명</t></is></c>'
            f'<c r="{category_column}1" t="inlineStr"><is><t>마이카테</t></is></c></row>'
        ]
        for row_number, (product_name, category) in enumerate(rows, start=2):
            xml_rows.append(
                f'<row r="{row_number}"><c r="{product_column}{row_number}" t="inlineStr"><is><t>{product_name}</t></is></c>'
                f'<c r="{category_column}{row_number}" t="inlineStr"><is><t>{category}</t></is></c></row>'
            )
        sheet = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<sheetData>{"".join(xml_rows)}</sheetData></worksheet>'
        )
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("xl/worksheets/sheet1.xml", sheet)


if __name__ == "__main__":
    unittest.main()
