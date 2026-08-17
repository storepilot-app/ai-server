import faiss

from app.category_matcher.config.settings import (
    EMBEDDING_PROVIDER,
    FAISS_NUM_THREADS,
    TORCH_NUM_INTEROP_THREADS,
    TORCH_NUM_THREADS,
)


def configure_cpu_runtime() -> None:
    faiss.omp_set_num_threads(FAISS_NUM_THREADS)
    if EMBEDDING_PROVIDER in {"local", "bge", "bge-m3"}:
        import torch

        torch.set_num_threads(TORCH_NUM_THREADS)
        torch.set_num_interop_threads(TORCH_NUM_INTEROP_THREADS)
