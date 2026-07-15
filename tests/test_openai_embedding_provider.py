import json
import unittest
from unittest.mock import patch

import numpy as np

from app.category_matcher.embedding import openai_provider
from app.category_matcher.embedding.openai_provider import OpenAIEmbeddingProvider


class OpenAIEmbeddingProviderTest(unittest.TestCase):
    def test_embeds_and_normalizes_vectors(self):
        response = _FakeResponse({
            "data": [
                {"index": 0, "embedding": [3.0, 4.0]},
                {"index": 1, "embedding": [0.0, 2.0]},
            ]
        })

        with patch.object(openai_provider, "EMBEDDING_API_KEY", "test-key"), patch.object(
            openai_provider.urllib.request, "urlopen", return_value=response
        ):
            vectors = OpenAIEmbeddingProvider().embed(["alpha", "beta"])

        self.assertEqual((2, 2), vectors.shape)
        self.assertTrue(np.allclose(vectors[0], np.asarray([0.6, 0.8], dtype=np.float32)))
        self.assertTrue(np.allclose(vectors[1], np.asarray([0.0, 1.0], dtype=np.float32)))


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


if __name__ == "__main__":
    unittest.main()
