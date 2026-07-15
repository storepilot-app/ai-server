from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    @property
    def provider_name(self) -> str:
        ...

    @property
    def model_name(self) -> str:
        ...

    def embed(self, texts: list[str]) -> np.ndarray:
        ...
