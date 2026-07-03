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

When product evidence does not meet the automatic acceptance policy, the AI server sends a hybrid set of up to five historical-product categories and five non-duplicate category-embedding candidates to an OpenAI-compatible chat completions API. The LLM selects one category or rejects all candidates.

Set these environment variables before running the server:

```env
STOREPILOT_LLM_API_KEY=your-api-key
STOREPILOT_LLM_MODEL=gpt-4o-mini
STOREPILOT_LLM_BASE_URL=https://api.openai.com/v1
STOREPILOT_LLM_TIMEOUT_SECONDS=90
STOREPILOT_LLM_BATCH_SIZE=15
STOREPILOT_LLM_MAX_CONCURRENCY=20
```

If `STOREPILOT_LLM_API_KEY` is empty, no automatic fallback category is selected. If there are no resolved similar products, the result is returned as `NO_SIMILAR_PRODUCTS` without an LLM call.
Ambiguous products are split into batches of 15 and up to 20 OpenAI requests run concurrently. Reduce `STOREPILOT_LLM_MAX_CONCURRENCY` if the provider returns rate-limit errors.

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

Builds the Naver category embedding cache used by hybrid product prediction.

```http
POST /ai/categories/predict
```

Returns an automatically accepted or LLM-selected category from historical-product and direct-category retrieval. Products without usable historical matches can still use category-embedding candidates.

## Category Alias Rules

Deterministic keyword-to-category mappings are managed in
`app/category_matcher/rules/category_aliases.csv`.

```csv
keyword,category_code,category_path
말랑이,50004246,출산/육아 > 완구/인형 > 미술놀이 > 클레이
```

Add one row per rule. `category_code` is used to resolve the current category cache;
`category_path` is documentation for people editing the file. Rules are reloaded for
each prediction request, and a longer keyword wins when multiple rules match.

## Historical Product Index

Historical product workbooks use column `D` for the product name and column `T` for the source user's my-category code. During rebuild, each my-category code is resolved to a Naver category ID, code, and full path. Only the resolved Naver category label is stored in the shared FAISS metadata.

The command-line rebuild requires a JSON array containing `myCategoryCode`, `categoryId`, `categoryCode`, and `fullPath`:

```powershell
uv run python -m scripts.rebuild_product_index `
  --mapping-json .\category-mappings.json `
  ..\List_20260627133027_uno1969_1.xlsx `
  ..\List_20260627133027_uno1969_2.xlsx `
  ..\List_20260627133027_uno1969_3.xlsx
```

The same rebuild is available through `POST /ai/categories/product-index/rebuild`. Spring Boot exposes the proxy API as `POST /api/v1/admin/training-products/rebuild`; its `userKey` is used only to resolve the source my-category codes while rebuilding. The resulting index is stored at `ai-cache/products/<model>/shared` and all users search the same index.

Prediction searches the historical index for 20 products, collapses near duplicates, and computes the category distribution from the complete set. It also searches the Naver category embedding cache, removes categories already present in the historical-product Top 5, and adds up to five direct-category candidates. A high-confidence historical-product consensus bypasses the LLM.

```env
STOREPILOT_AUTO_ACCEPT_THRESHOLD=0.90
STOREPILOT_LLM_THRESHOLD=0.85
STOREPILOT_CATEGORY_SUPPORT_THRESHOLD=0.75
STOREPILOT_CATEGORY_MARGIN_THRESHOLD=0.15
STOREPILOT_AUTO_ACCEPT_MIN_EXAMPLES=3
STOREPILOT_PRODUCT_DUPLICATE_THRESHOLD=0.985
STOREPILOT_PRODUCT_SEARCH_K=20
STOREPILOT_PRODUCT_REPRESENTATIVE_K=5
STOREPILOT_CATEGORY_EMBEDDING_SEARCH_K=15
STOREPILOT_CATEGORY_EMBEDDING_CANDIDATE_K=5
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
