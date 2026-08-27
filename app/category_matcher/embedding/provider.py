from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    @property
    def provider_name(self) -> str:
        ...

    @property
    def model_name(self) -> str:
        ...

    @property
    def cache_key(self) -> str:
        ...

    @property
    def is_asymmetric(self) -> bool:
        ...

    def embed(self, texts: list[str]) -> np.ndarray:
        ...

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        ...

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        ...
