import json
import unittest
from unittest.mock import patch

import numpy as np

from app.category_matcher.embedding import solar_provider
from app.category_matcher.embedding.solar_provider import SolarEmbeddingProvider


class SolarEmbeddingProviderTest(unittest.TestCase):
    def test_uses_query_model_and_normalizes_vectors(self):
        response = _FakeResponse({
            "data": [
                {"index": 0, "embedding": [3.0, 4.0]},
                {"index": 1, "embedding": [0.0, 2.0]},
            ]
        })

        with patch.object(solar_provider, "SOLAR_EMBEDDING_API_KEY", "test-key"), patch.object(
            solar_provider, "SOLAR_EMBEDDING_DIMENSIONS", 2
        ), patch.object(
            solar_provider, "SOLAR_EMBEDDING_REQUESTS_PER_MINUTE", 0
        ), patch.object(solar_provider.urllib.request, "urlopen", return_value=response) as urlopen:
            vectors = SolarEmbeddingProvider().embed_queries(["alpha", "beta"])

        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual("solar-embedding-2-query", payload["model"])
        self.assertNotIn("dimensions", payload)
        self.assertEqual((2, 2), vectors.shape)
        self.assertTrue(np.allclose(vectors[0], np.asarray([0.6, 0.8], dtype=np.float32)))

    def test_uses_passage_model(self):
        response = _FakeResponse({"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

        with patch.object(solar_provider, "SOLAR_EMBEDDING_API_KEY", "test-key"), patch.object(
            solar_provider, "SOLAR_EMBEDDING_DIMENSIONS", 2
        ), patch.object(
            solar_provider, "SOLAR_EMBEDDING_REQUESTS_PER_MINUTE", 0
        ), patch.object(solar_provider.urllib.request, "urlopen", return_value=response) as urlopen:
            SolarEmbeddingProvider().embed_passages(["document"])

        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual("solar-embedding-2-passage", payload["model"])

    def test_spaces_requests_to_stay_under_rpm_limit(self):
        solar_provider._next_request_at = 0.0

        with patch.object(
            solar_provider, "SOLAR_EMBEDDING_REQUESTS_PER_MINUTE", 60
        ), patch.object(
            solar_provider.time, "monotonic", side_effect=[100.0, 100.0, 100.0, 101.0]
        ), patch.object(solar_provider.time, "sleep") as sleep:
            solar_provider._wait_for_rate_limit()
            solar_provider._wait_for_rate_limit()

        sleep.assert_called_once_with(1.0)


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
