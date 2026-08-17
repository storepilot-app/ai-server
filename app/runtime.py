import faiss
import torch

from app.category_matcher.config.settings import (
    FAISS_NUM_THREADS,
    TORCH_NUM_INTEROP_THREADS,
    TORCH_NUM_THREADS,
)


def configure_cpu_runtime() -> None:
    torch.set_num_threads(TORCH_NUM_THREADS)
    torch.set_num_interop_threads(TORCH_NUM_INTEROP_THREADS)
    faiss.omp_set_num_threads(FAISS_NUM_THREADS)
