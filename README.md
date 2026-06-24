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

The server automatically loads `ai-server/.env` on startup. Real environment variables take priority over values in the file.

```powershell
$env:UV_CACHE_DIR="C:\Project\StorePilot\ai-server\.uv-cache"
$env:UV_PYTHON_INSTALL_DIR="C:\Project\StorePilot\ai-server\.uv-python"
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## LLM Category Judge

After BGE-M3 returns Top-5 category candidates, the AI server can call an OpenAI-compatible chat completions API for every product and ask the LLM to select the best candidate or reject all candidates.

Set these environment variables before running the server:

```env
STOREPILOT_LLM_API_KEY=your-api-key
STOREPILOT_LLM_MODEL=gpt-4o-mini
STOREPILOT_LLM_BASE_URL=https://api.openai.com/v1
STOREPILOT_LLM_TIMEOUT_SECONDS=20
```

If `STOREPILOT_LLM_API_KEY` is empty, the server skips the LLM call and uses the embedding Top-1 result. If the LLM rejects all Top-5 candidates, the final category is returned as no match while the Top-5 candidates remain available for Excel review.

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
