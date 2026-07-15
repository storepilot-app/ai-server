from app.category_matcher.config.settings import EMBEDDING_PROVIDER
from app.category_matcher.embedding.local_provider import LocalEmbeddingProvider
from app.category_matcher.embedding.openai_provider import OpenAIEmbeddingProvider
from app.category_matcher.embedding.provider import EmbeddingProvider


_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    global _provider
    if _provider is None:
        _provider = _create_embedding_provider()
    return _provider


def _create_embedding_provider() -> EmbeddingProvider:
    if EMBEDDING_PROVIDER in {"local", "bge", "bge-m3"}:
        return LocalEmbeddingProvider()
    if EMBEDDING_PROVIDER in {"openai", "api"}:
        return OpenAIEmbeddingProvider()
    raise ValueError(f"Unsupported embedding provider: {EMBEDDING_PROVIDER}")
