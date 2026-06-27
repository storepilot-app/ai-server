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

After BGE-M3 returns Top-10 category candidates, the AI server can call an OpenAI-compatible chat completions API for every product and ask the LLM to select the best candidate or reject all candidates.

Set these environment variables before running the server:

```env
STOREPILOT_LLM_API_KEY=your-api-key
STOREPILOT_LLM_MODEL=gpt-4o-mini
STOREPILOT_LLM_BASE_URL=https://api.openai.com/v1
STOREPILOT_LLM_TIMEOUT_SECONDS=90
STOREPILOT_LLM_BATCH_SIZE=30
```

If `STOREPILOT_LLM_API_KEY` is empty, the server skips the LLM call and uses the embedding Top-1 result. If the LLM rejects all Top-10 candidates, the final category is returned as no match while the Top-10 candidates remain available for Excel review.

## Embedding Model

Default model:

```text
BAAI/bge-m3
```

CUDA is selected automatically when a CUDA-enabled PyTorch build is installed. Override it with:

```env
STOREPILOT_EMBEDDING_DEVICE=auto
STOREPILOT_EMBEDDING_BATCH_SIZE=32
STOREPILOT_EMBEDDING_USE_FP16=true
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

## Historical Product Index

Historical product workbooks use column `D` for the product name and column `T` for the user's my-category code. Build the initial index for the three workbooks in the StorePilot root with:

```powershell
uv run python -m scripts.rebuild_product_index `
  --user-key uno1969 `
  ..\List_20260627133027_uno1969_1.xlsx `
  ..\List_20260627133027_uno1969_2.xlsx `
  ..\List_20260627133027_uno1969_3.xlsx
```

The same rebuild is available through `POST /ai/categories/product-index/rebuild`. Spring Boot exposes the proxy API as `POST /api/v1/admin/training-products/rebuild`.

Prediction searches the historical index for 20 products, collapses near duplicates, computes the category distribution from the complete set, and sends a category-diverse Top 5 to the LLM. A high-confidence consensus bypasses the LLM.

```env
STOREPILOT_AUTO_ACCEPT_THRESHOLD=0.95
STOREPILOT_LLM_THRESHOLD=0.85
STOREPILOT_CATEGORY_SUPPORT_THRESHOLD=0.75
STOREPILOT_CATEGORY_MARGIN_THRESHOLD=0.15
STOREPILOT_AUTO_ACCEPT_MIN_EXAMPLES=3
STOREPILOT_PRODUCT_DUPLICATE_THRESHOLD=0.985
STOREPILOT_PRODUCT_SEARCH_K=20
STOREPILOT_PRODUCT_REPRESENTATIVE_K=5
STOREPILOT_PRODUCT_MAX_PER_CATEGORY=2
```

`flat` is the default exact index and is appropriate for the current data size. Rebuild with HNSW when the collection grows to several hundred thousand products:

```env
STOREPILOT_PRODUCT_INDEX_TYPE=hnsw
STOREPILOT_PRODUCT_HNSW_M=32
STOREPILOT_PRODUCT_HNSW_EF_CONSTRUCTION=200
STOREPILOT_PRODUCT_HNSW_EF_SEARCH=64
```

User corrections are persisted by Spring Boot and immediately appended to FAISS through `POST /api/v1/admin/training-products/feedback`.

## Spring Boot Integration

Spring Boot calls the AI server at:

```text
storepilot.ai.base-url=http://127.0.0.1:8000
```

If the AI cache is missing, Spring Boot tries to rebuild embeddings and retries prediction once.
