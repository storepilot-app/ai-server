# StorePilot AI Server

FastAPI server for matching product names to Naver categories with `BAAI/bge-m3`.

## Python

The AI server uses PyTorch through `sentence-transformers`, so it is pinned to Python 3.12.

```text
requires-python = ">=3.12,<3.13"
```

## Setup With uv

On this Windows project, keep uv cache and managed Python paths inside the project to avoid user-directory permission issues.

```powershell
cd C:\Project\StorePilot\ai-server
$env:UV_CACHE_DIR="C:\Project\StorePilot\ai-server\.uv-cache"
$env:UV_PYTHON_INSTALL_DIR="C:\Project\StorePilot\ai-server\.uv-python"
uv sync
```

If uv cannot discover its managed Python, point uv at an existing Python 3.12 interpreter:

```powershell
uv sync --python C:\Path\To\Python312\python.exe
```

## Run

```powershell
$env:UV_CACHE_DIR="C:\Project\StorePilot\ai-server\.uv-cache"
$env:UV_PYTHON_INSTALL_DIR="C:\Project\StorePilot\ai-server\.uv-python"
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Embedding Model

Default model:

```text
BAAI/bge-m3
```

The cache is separated by model name, so old `multilingual-e5-small` embeddings will not be reused with BGE-M3.

To temporarily switch models:

```powershell
$env:STOREPILOT_EMBEDDING_MODEL="intfloat/multilingual-e5-small"
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## API

```http
POST /ai/categories/rebuild
```

Builds category embeddings for a Naver category version.

```http
POST /ai/categories/predict
```

Returns the Top 1 Naver category for each product name.

## Spring Boot Integration

Spring Boot calls the AI server at:

```text
storepilot.ai.base-url=http://127.0.0.1:8000
```

If the AI cache is missing, Spring Boot tries to rebuild embeddings and retries prediction once.
