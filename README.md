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

When product evidence does not meet the automatic acceptance policy, the AI server sends the category-diverse similar-product Top 5 to an OpenAI-compatible chat completions API. The LLM selects one of those product categories or rejects all candidates.

Set these environment variables before running the server:

```env
STOREPILOT_LLM_API_KEY=your-api-key
STOREPILOT_LLM_MODEL=gpt-4o-mini
STOREPILOT_LLM_BASE_URL=https://api.openai.com/v1
STOREPILOT_LLM_TIMEOUT_SECONDS=90
STOREPILOT_LLM_BATCH_SIZE=30
```

If `STOREPILOT_LLM_API_KEY` is empty, no automatic fallback category is selected. If there are no resolved similar products, the result is returned as `NO_SIMILAR_PRODUCTS` without an LLM call.

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

Builds legacy Naver category embeddings. Product prediction no longer uses this cache.

```http
POST /ai/categories/predict
```

Returns an automatically accepted or LLM-selected category from the historical product index. Products without usable similar products return no match.

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
