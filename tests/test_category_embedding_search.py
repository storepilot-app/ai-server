import unittest
from unittest.mock import patch

import numpy as np

from app.category_matcher.embedding import store


class CategoryEmbeddingSearchTest(unittest.TestCase):
    def test_returns_highest_scoring_categories_in_order(self) -> None:
        category_embeddings = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.2],
            ],
            dtype=np.float32,
        )
        categories = [
            {"categoryId": 1, "categoryCode": "A", "fullPath": "Category A"},
            {"categoryId": 2, "categoryCode": "B", "fullPath": "Category B"},
            {"categoryId": 3, "categoryCode": "C", "fullPath": "Category C"},
        ]
        query_vectors = np.asarray([[1.0, 0.0]], dtype=np.float32)

        with patch.object(
            store,
            "load_category_cache",
            return_value=(category_embeddings, categories),
        ), patch.object(store, "CATEGORY_EMBEDDING_SEARCH_K", 2):
            results = store.search_category_candidates_by_vectors(1, query_vectors)

        self.assertEqual([1, 3], [candidate.categoryId for candidate in results[0]])
        self.assertAlmostEqual(1.0, results[0][0].score)
        self.assertAlmostEqual(0.8, results[0][1].score)

    def test_returns_empty_candidates_when_cache_is_missing(self) -> None:
        query_vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        with patch.object(
            store,
            "load_category_cache",
            return_value=(np.empty((0, 0), dtype=np.float32), []),
        ):
            results = store.search_category_candidates_by_vectors(1, query_vectors)

        self.assertEqual([[], []], results)


if __name__ == "__main__":
    unittest.main()
