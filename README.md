# StorePilot AI Server

StorePilot의 FastAPI 기반 AI 서버입니다. 네이버 카테고리 임베딩, 기존 상품 임베딩, FAISS 인덱스를 생성하고 검색합니다.

현재 기본 임베딩 설정은 로컬 `BAAI/bge-m3`입니다.

## 기술 스택

- Python 3.12
- FastAPI
- Uvicorn
- sentence-transformers
- PyTorch
- FAISS CPU
- NumPy

## 담당 기능

- 네이버 카테고리 임베딩 캐시 생성
- 상품명 기반 네이버 카테고리 예측
- 기존 상품 엑셀 기반 FAISS 인덱스 생성
- 유사 기존 상품 검색
- 임베딩/기존 상품 근거가 애매한 경우 LLM 보조 판단
- 카테고리 보정 피드백을 기존 상품 인덱스에 반영

## Python 버전

PyTorch와 sentence-transformers 실행 환경 때문에 Python 3.12를 사용합니다.

```text
requires-python = ">=3.12,<3.13"
```

## uv로 설치

Windows에서는 uv 캐시와 관리 Python 경로를 프로젝트 내부에 두면 권한 문제를 줄일 수 있습니다.

```powershell
cd C:\Project\StorePilot\ai-server
$env:UV_CACHE_DIR="C:\Project\StorePilot\ai-server\.uv-cache"
$env:UV_PYTHON_INSTALL_DIR="C:\Project\StorePilot\ai-server\.uv-python"
uv sync
```

uv가 관리 Python을 찾지 못하는 경우:

```powershell
uv sync --python C:\Path\To\Python312\python.exe
```

## 환경변수

`.env.example`을 `.env`로 복사합니다.

서버는 `app.main`을 통해 실행될 때 `ai-server/.env`를 자동으로 읽습니다. 단, 실제 프로세스 환경변수가 이미 설정되어 있으면 `.env`보다 우선합니다.

기본 권장 설정은 OpenAI 임베딩 API입니다.

```env
STOREPILOT_EMBEDDING_PROVIDER=openai
STOREPILOT_EMBEDDING_API_KEY=
STOREPILOT_EMBEDDING_API_BASE_URL=https://api.openai.com/v1
STOREPILOT_EMBEDDING_API_MODEL=text-embedding-3-small
STOREPILOT_EMBEDDING_API_DIMENSIONS=1536
STOREPILOT_EMBEDDING_API_TIMEOUT_SECONDS=60
STOREPILOT_EMBEDDING_API_BATCH_SIZE=512
```

외부 API 대신 로컬 BGE-M3를 사용하려면:

```env
STOREPILOT_EMBEDDING_PROVIDER=local
STOREPILOT_EMBEDDING_MODEL=BAAI/bge-m3
STOREPILOT_EMBEDDING_DEVICE=cpu
STOREPILOT_EMBEDDING_BATCH_SIZE=32
STOREPILOT_EMBEDDING_USE_FP16=false
```

## 실행

```powershell
cd C:\Project\StorePilot\ai-server
$env:UV_CACHE_DIR="C:\Project\StorePilot\ai-server\.uv-cache"
$env:UV_PYTHON_INSTALL_DIR="C:\Project\StorePilot\ai-server\.uv-python"
uv run python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

헬스 체크:

```text
http://127.0.0.1:8000/health
```

## CPU/GPU 동작 확인

아래 명령은 `app.main`을 통해 `.env`를 먼저 로드한 뒤 현재 임베딩 장치를 확인합니다.

```powershell
uv run python -c "import app.main; from app.category_matcher.config.settings import EMBEDDING_DEVICE; from app.category_matcher.embedding.local_provider import LocalEmbeddingProvider; p=LocalEmbeddingProvider(); print(EMBEDDING_DEVICE); print(p._resolve_device())"
```

CPU 설정이 정상 적용되면:

```text
cpu
cpu
```

## 임베딩 Provider

AI 서버는 Provider 구조로 임베딩 방식을 선택합니다.

- `openai`: OpenAI 호환 embeddings API 사용, 현재 기본값
- `local`: 로컬 sentence-transformers 모델 사용, 기본 로컬 모델은 `BAAI/bge-m3`

OpenAI 호환 임베딩 API 설정:

```env
STOREPILOT_EMBEDDING_PROVIDER=openai
STOREPILOT_EMBEDDING_API_KEY=
STOREPILOT_EMBEDDING_API_BASE_URL=https://api.openai.com/v1
STOREPILOT_EMBEDDING_API_MODEL=text-embedding-3-small
STOREPILOT_EMBEDDING_API_DIMENSIONS=1536
STOREPILOT_EMBEDDING_API_TIMEOUT_SECONDS=60
STOREPILOT_EMBEDDING_API_BATCH_SIZE=512
```

OpenAI 임베딩 API 응답의 토큰 사용량과 응답시간은 `embedding_api_timing` 로그로 확인할 수 있습니다.

## 캐시 구조

런타임 캐시는 아래 디렉터리에 저장됩니다.

```text
ai-cache/
```

주요 경로:

```text
ai-cache/categories/<provider-model>/version-<versionId>/
ai-cache/products/<provider-model>/shared/
```

캐시는 provider/model/dimensions 기준으로 분리됩니다. 따라서 BGE-M3 캐시와 OpenAI 임베딩 캐시는 서로 덮어쓰지 않습니다.

임베딩 provider 또는 모델을 바꾸는 경우:

1. AI 서버를 재시작합니다.
2. 네이버 카테고리 임베딩 캐시를 새 provider/model로 생성합니다.
3. 기존 상품 FAISS 인덱스도 같은 provider/model로 다시 생성합니다.

오래된 캐시 삭제는 필수가 아닙니다. 디스크 용량을 줄이거나 혼동을 피하고 싶을 때만 삭제하면 됩니다.

## 메모리 참고

BGE-M3는 서버 시작 즉시 로드되는 것이 아니라 첫 임베딩 요청 시점에 로드됩니다.

일반적인 로컬 동작:

- 서버 시작 직후: 상대적으로 낮은 메모리 사용량
- 첫 카테고리/상품 예측 실행: BGE-M3, tokenizer, FAISS 인덱스, 임베딩 배열이 RAM에 로드됨
- 이후 메모리가 내려가지 않고 유지됨: 다음 요청에서 재사용하기 위한 정상 동작

운영 서버 메모리는 시작 직후 사용량이 아니라 첫 요청 이후 유지되는 메모리 기준으로 잡아야 합니다.

`be + ai-server`를 같은 EC2에 배포한다면 최소 4GB RAM을 권장하고, swap 설정도 고려합니다.

## API

```http
GET /health
```

```http
POST /ai/categories/rebuild
```

카테고리 버전에 대한 네이버 카테고리 임베딩 캐시를 생성합니다.

```http
POST /ai/categories/predict
```

기존 상품 검색, 직접 카테고리 검색, alias rule, 선택적 LLM fallback을 조합해 카테고리를 예측합니다.

```http
POST /ai/categories/product-index/rebuild
```

기존 상품 엑셀 파일로부터 공유 FAISS 상품 인덱스를 생성합니다.

```http
POST /ai/categories/product-index/feedback
```

보정된 상품/카테고리 쌍을 기존 상품 인덱스에 추가합니다.

## 카테고리 Alias Rule

확정 키워드 기반 카테고리 매핑은 아래 파일에서 관리합니다.

```text
app/category_matcher/rules/category_aliases.csv
```

형식:

```csv
keyword,category_code,category_path
말랑이,50004246,출산/육아 > 완구/인형 > 미술놀이 > 클레이
```

Rule은 예측 요청마다 다시 읽습니다. 여러 키워드가 동시에 매칭되면 더 긴 키워드를 우선합니다.

## 기존 상품 인덱스

기존 상품 엑셀은 현재 아래 컬럼을 사용합니다.

- `D` 컬럼: 상품명
- `T` 컬럼: 원본 사용자의 내 카테고리 코드

인덱스 재생성 시 Spring Boot가 사용자의 내 카테고리 매핑을 AI 서버로 전달합니다. AI 서버는 내 카테고리 코드를 네이버 카테고리 라벨로 변환한 뒤, 변환된 네이버 카테고리 정보만 공유 상품 인덱스에 저장합니다.

현재 기본 검색 설정:

```env
STOREPILOT_PRODUCT_INDEX_TYPE=flat
STOREPILOT_PRODUCT_SEARCH_K=20
STOREPILOT_PRODUCT_REPRESENTATIVE_K=5
STOREPILOT_PRODUCT_DUPLICATE_THRESHOLD=0.985
STOREPILOT_CATEGORY_EMBEDDING_SEARCH_K=15
STOREPILOT_CATEGORY_EMBEDDING_CANDIDATE_K=5
```

`flat`은 전체 벡터를 정확 검색하는 기본 인덱스입니다. 현재 데이터 규모에서는 적합합니다.

데이터가 수십만 건 이상으로 커지면 HNSW 옵션을 사용할 수 있습니다.

```env
STOREPILOT_PRODUCT_INDEX_TYPE=hnsw
STOREPILOT_PRODUCT_HNSW_M=32
STOREPILOT_PRODUCT_HNSW_EF_CONSTRUCTION=200
STOREPILOT_PRODUCT_HNSW_EF_SEARCH=64
```

## LLM Fallback

기존 상품 근거만으로 자동 판단하기 애매한 경우 OpenAI 호환 chat completions API를 호출할 수 있습니다.

```env
STOREPILOT_LLM_API_KEY=
STOREPILOT_LLM_MODEL=gpt-4o-mini
STOREPILOT_LLM_BASE_URL=https://api.openai.com/v1
STOREPILOT_LLM_TIMEOUT_SECONDS=90
STOREPILOT_LLM_BATCH_SIZE=15
STOREPILOT_LLM_MAX_CONCURRENCY=5
```

`STOREPILOT_LLM_API_KEY`가 비어 있으면 LLM fallback은 실행되지 않습니다.

## CPU 동시 실행 제한

CPU에서 로컬 임베딩 모델과 FAISS를 실행하는 배포 서버는 다음 값으로 연산 스레드와
동시 작업 수를 제한할 수 있습니다.

```env
STOREPILOT_TORCH_NUM_THREADS=2
STOREPILOT_TORCH_NUM_INTEROP_THREADS=1
STOREPILOT_FAISS_NUM_THREADS=2
STOREPILOT_AI_MAX_CONCURRENT_CPU_TASKS=2
```

기본값은 물리 4코어 서버에서 CPU 작업 2개가 각각 연산 스레드 2개를 사용하는 구성을
기준으로 합니다. `STOREPILOT_AI_MAX_CONCURRENT_CPU_TASKS`를 초과한 예측·인덱스 생성
요청은 실행 중인 CPU 작업이 끝날 때까지 AI 서버 내부에서 대기합니다. Uvicorn worker는
모델 메모리 중복과 CPU 경합을 피하기 위해 1개로 실행하는 것을 권장합니다.

## 테스트

```powershell
uv run python -m unittest discover tests
uv run python -m compileall app tests
```

## Spring Boot 연동

Spring Boot 백엔드는 아래 주소로 AI 서버를 호출합니다.

```env
AI_SERVER_BASE_URL=http://127.0.0.1:8000
```

백엔드와 같은 EC2에 배포하는 경우 AI 서버는 `127.0.0.1`에만 바인딩해서 외부에 직접 노출하지 않는 것을 권장합니다.
