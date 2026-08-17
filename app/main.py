import os
from pathlib import Path

from fastapi import FastAPI


def load_project_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


load_project_env()

def configure_native_thread_environment() -> None:
    thread_count = os.getenv("STOREPILOT_TORCH_NUM_THREADS", "2")
    for variable_name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        os.environ[variable_name] = thread_count


configure_native_thread_environment()

from app.runtime import configure_cpu_runtime

configure_cpu_runtime()

from app.category_matcher.router import router as category_matcher_router

app = FastAPI(title="StorePilot AI Server")
app.include_router(category_matcher_router, prefix="/ai/categories", tags=["Category Matcher"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
