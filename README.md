# StorePilot AI Server

상품명과 기존 상품의 카테고리 데이터를 기반으로 네이버 쇼핑 카테고리를 추천하는 FastAPI 서버입니다.
기존 상품 유사도 검색과 카테고리 자체 검색을 결합하고, 근거가 충분하면 자동 선택하며 애매한 경우 LLM이 후보를 판단합니다.

여기서 **학습은 모델 파인튜닝이 아니라 상품명–카테고리 데이터를 임베딩하여 검색 인덱스에 반영하는 과정**입니다.
웹페이지나 쇼핑 검색 API를 상품마다 조회하지 않습니다. 다만 기본 설정에서는 임베딩 및 LLM 판단에 외부 API를 사용합니다.

## 1. 시스템 역할

```text
사용자 → Frontend → Spring Boot BE → AI Server
                     │                 ├─ Solar / OpenAI / 로컬 임베딩
                     │                 ├─ 기존 상품: FAISS 검색
                     │                 ├─ 카테고리: NumPy 벡터 검색
                     │                 └─ 자동 선택 / Alias / LLM 판단
                     └─ 사용자별 마이카테 매핑·키워드 생성·결과 엑셀
```

| 구분 | 담당 |
| --- | --- |
| AI 서버 | 카테고리 임베딩 생성, 공유 상품 인덱스 관리, 네이버 카테고리 예측 및 근거 반환 |
| BE | 인증·권한, 사용자별 마이카테 매핑, 업로드 검증, 비동기 엑셀 작업, 사용량 제한 |
| BE | 규칙 기반 상품 키워드 생성, 결과 엑셀 및 이미지 다운로드 |

AI 상품 인덱스는 **사용자별 인덱스가 아닌 공유 인덱스**입니다. 업로드자의 마이카테 코드를 네이버 카테고리로 변환해서 저장하고, 예측 결과를 각 사용자의 마이카테로 변환하는 일은 BE에서 수행합니다.

## 2. 기술 스택

[pyproject.toml](pyproject.toml) 기준입니다. 실제 설치 버전은 [uv.lock](uv.lock)으로 관리합니다.

| 기술 | 설정 | 용도 |
| --- | --- | --- |
| Python | 3.12 (`>=3.12,<3.13`) | 실행 환경 |
| FastAPI / Pydantic | 0.115.6 / 2.10.4 | HTTP API·요청/응답 스키마 |
| Uvicorn | 0.34.0 | ASGI 서버 |
| NumPy | 2.2.1 | 벡터 정규화·카테고리 유사도 계산 |
| FAISS CPU | `>=1.14.3` | 상품 벡터 검색 |
| sentence-transformers / PyTorch | 3.3.1 / `>=2.11.0` | 로컬 임베딩 |
| python-multipart | `>=0.0.32` | 엑셀 업로드 수신 |

외부 임베딩·LLM API는 Python 표준 라이브러리 `urllib.request`로 호출합니다.
현재 uv 설정은 PyTorch를 CUDA 12.8 인덱스에서 설치합니다. Solar 사용 시 GPU는 필요하지 않지만, 설치 의존성에 로컬 모델용 라이브러리도 포함되어 있습니다.

## 3. 디렉터리 구조

```text
app/
├─ main.py                         # .env 로드, FastAPI 생성, 헬스 체크
└─ category_matcher/
   ├─ router.py                    # 카테고리·상품 인덱스 API
   ├─ schemas.py                   # 요청/응답 모델
   ├─ service.py                   # 예측 흐름 조합 및 단계별 로그
   ├─ config/settings.py           # 환경변수와 기본값
   ├─ preprocess/query.py          # 상품명 전처리
   ├─ embedding/
   │  ├─ provider.py / factory.py  # 임베딩 역할 및 구현 선택
   │  ├─ solar_provider.py         # Solar query/passage, 요청 간격 제한
   │  ├─ openai_provider.py        # OpenAI 호환 임베딩
   │  ├─ local_provider.py         # sentence-transformers
   │  └─ store.py                  # 버전별 카테고리 캐시·검색
   ├─ product_memory/
   │  ├─ excel.py                  # xlsx ZIP/XML 해석
   │  └─ store.py                  # 공유 FAISS 인덱스·중복·피드백
   ├─ retrieval/evidence.py        # 카테고리 지지도·대표 상품 구성
   ├─ decision/policy.py           # 자동 선택 조건
   ├─ rules/                      # 키워드 기반 카테고리 Alias
   └─ llm/judge.py                 # 후보 구성·LLM 배치 판단
scripts/rebuild_product_index.py   # 상품 인덱스 재생성 CLI
tests/                            # unittest 기반 테스트
```

## 4. 카테고리 예측 흐름

구현: [service.py](app/category_matcher/service.py)

1. **상품명 전처리**: Unicode NFKC 정규화, 괄호 기호·구분자·일부 수량 표현을 정리합니다. 괄호 안 의미 있는 텍스트는 유지합니다.
2. **임베딩 생성**: 입력 목록을 provider의 배치 크기에 따라 처리하고 정규화된 벡터를 생성합니다.
3. **기존 상품 검색**: FAISS로 기본 최대 60건을 먼저 검색하고, 벡터가 거의 같은 검색 결과를 제외하여 최대 20건을 남깁니다.
4. **카테고리 직접 검색**: 요청한 `versionId`의 카테고리 행렬과 내적을 계산해 상위 15개를 검색합니다.
5. **근거 구성**: 충돌 라벨 상품을 제외하고 유사도 가중 지지도를 계산합니다. 상품 근거의 상위 카테고리 최대 5개와, 이들과 중복되지 않는 직접 검색 후보 최대 5개를 구성합니다.
6. **결정**: Alias 규칙을 우선 적용하고, 상품 근거가 자동 선택 조건을 만족하면 LLM 호출을 생략합니다. 나머지는 LLM에 후보 선택 또는 거절을 요청합니다.
7. **결과 반환**: 입력 순서대로 `rowId`, 선택 카테고리, 유사 상품, 지지도, LLM 상태 등을 반환합니다.

### 대칭 검색과 비대칭 검색

| 검색 대상 | Solar 입력 상품 벡터 | 저장된 벡터 |
| --- | --- | --- |
| 기존 상품 | passage | passage |
| 네이버 카테고리 | query | passage |

Solar는 기본적으로 상품명을 두 역할로 임베딩합니다. 두 호출은 현재 순차 실행합니다.
로컬 및 OpenAI provider는 query/passage 역할에 같은 구현을 사용하므로 한 번 만든 상품 벡터를 두 검색에서 재사용합니다.

### 자동 선택과 LLM

[자동 선택 정책](app/category_matcher/decision/policy.py)은 아래 조건을 **모두** 만족해야 합니다.

| 환경변수 | 기본값 | 의미 |
| --- | --- | --- |
| `STOREPILOT_AUTO_ACCEPT_THRESHOLD` | 0.90 | 1위 카테고리의 최대 상품 유사도 |
| `STOREPILOT_CATEGORY_SUPPORT_THRESHOLD` | 0.75 | 1위 카테고리의 가중 지지도 |
| `STOREPILOT_CATEGORY_MARGIN_THRESHOLD` | 0.15 | 1위와 2위의 지지도 차이 |
| `STOREPILOT_AUTO_ACCEPT_MIN_EXAMPLES` | 3 | 1위 카테고리를 지지하는 상품 수 |

지지도는 유사도에 지수 가중치를 적용한 상대 비율이며, 정답 확률이 아닙니다. 응답의 `score` 역시 임베딩 유사도이지 검증된 정확도 지표가 아닙니다.

LLM은 기본 15개 상품씩 묶어 요청하고, 한 예측 호출 안에서 최대 20개 배치를 병렬 실행합니다.
키 미설정·호출 실패·응답 누락 시에는 **첫 번째 후보로 대체**합니다. LLM이 명시적으로 거절하면 카테고리를 선택하지 않습니다.
따라서 HTTP 성공 여부뿐 아니라 `llmUsed`, `llmStatus`, `llmStatusDetail`을 함께 확인해야 합니다.

`STOREPILOT_LLM_THRESHOLD`는 설정에 남아 있지만 현재 판단 흐름에서 사용하지 않습니다.

Alias는 [category_aliases.csv](app/category_matcher/rules/category_aliases.csv)에 `keyword,category_code,category_path` 형식으로 관리합니다.
요청마다 읽고, 여러 규칙이 일치하면 긴 키워드를 우선합니다. 현재 흐름에서는 Alias 판단 전에도 임베딩과 검색을 수행합니다.

## 5. 상품 데이터 반영과 중복 처리

### 전체 재생성

[Excel 파서](app/category_matcher/product_memory/excel.py)는 `.xlsx`의 워크시트 XML을 순회하며 각 시트의 첫 행에서 아래 헤더를 찾습니다. **D/T 고정열 방식이 아닙니다.**

| 데이터 | 허용 헤더 |
| --- | --- |
| 상품명 | `상품명` |
| 마이카테 코드 | `마이카테`, `마이카테고리`, `마이카테고리코드` |

헤더 비교 시 공백을 제거합니다. 상품명과 마이카테 값이 모두 있는 행을 읽고, 요청에 포함된 매핑으로 네이버 카테고리를 찾습니다. 매핑되지 않은 데이터는 인덱스에 넣지 않습니다.

- 상품명을 소문자화하고 한글·영문·숫자만 남긴 정규화 키로 중복을 합칩니다.
- 동일 키에 여러 카테고리가 있으면 라벨을 모두 보관하고 충돌 상품으로 집계합니다.
- 충돌 상품은 검색 인덱스에는 남지만 카테고리 근거 계산에서 제외합니다.
- 상품 목록을 512개 단위로 provider에 전달합니다. 실제 API 요청은 provider의 배치 크기로 다시 나뉩니다.
- 재생성은 **전달한 파일들로 공유 상품 인덱스 전체를 교체**합니다. 특정 사용자의 데이터만 교체하는 기능이 아닙니다.

반환 통계의 `sourceRowCount`는 빈 행을 포함한 엑셀 전체 행 수가 아니라 파서가 반환한 행 수입니다.
`duplicateRowCount`는 유효 행 수에서 정규화 후 상품 수를 뺀 값입니다.

### 추가 및 보정 피드백

[상품 인덱스 저장소](app/category_matcher/product_memory/store.py)의 단건·배치 피드백 API를 사용합니다.

- 새 정규화 상품명: passage 벡터를 생성해 추가합니다.
- 기존 정규화 상품명: 벡터를 재생성하지 않고 상품 메타데이터와 카테고리를 교체합니다.
- 동일 상품의 기존 복수 라벨은 전달받은 단일 라벨로 교체됩니다.
- 한 배치에 같은 정규화 상품명이 반복되면 마지막 항목이 적용됩니다.

재생성 시의 중복 병합과 검색 시의 유사 벡터 제거는 별도 처리입니다.
`STOREPILOT_PRODUCT_DUPLICATE_THRESHOLD=0.985`는 검색 결과 다양성을 위한 기준이며, 인덱스 저장 단계의 문자열 중복 판정 기준이 아닙니다.

## 6. API

라우트: [router.py](app/category_matcher/router.py) · 상세 스키마: [schemas.py](app/category_matcher/schemas.py)

| 메서드 | 경로 | 요청 | 역할 |
| --- | --- | --- | --- |
| GET | `/health` | 없음 | 프로세스 응답 확인 |
| POST | `/ai/categories/rebuild` | JSON: `versionId`, `categories` | 버전별 카테고리 임베딩 생성 |
| POST | `/ai/categories/predict` | JSON: `versionId`, `products` | 카테고리 예측 |
| POST | `/ai/categories/product-index/rebuild` | multipart: `userId`, `categoryMappings`, `files` | 공유 상품 인덱스 전체 재생성 |
| POST | `/ai/categories/product-index/feedback` | JSON: 상품명·네이버 카테고리·`userId` | 단건 추가/보정 |
| POST | `/ai/categories/product-index/feedback/batch` | JSON: `userId`, `products` | 여러 상품 추가/보정 |

`categoryMappings`는 `myCategoryCode`, `categoryId`, `categoryCode`, `fullPath` 객체 배열을 JSON 문자열로 전달합니다.
`userId`는 공유 인덱스의 저장 경로나 검색 필터로 사용하지 않습니다.

예측 요청 예시:

```json
{
  "versionId": 8,
  "products": [
    {"rowId": 2, "productName": "린넨 반팔 셔츠"},
    {"rowId": 3, "productName": "스테인리스 주방 집게"}
  ]
}
```

`versionId`는 실제 생성한 카테고리 버전을 사용하고, 한 요청 안의 `rowId`는 중복되지 않게 전달해야 합니다.
상품 재생성·피드백 라우트는 일부 입력 오류를 400, 외부 임베딩 호출의 RuntimeError를 502로 변환합니다.
요청 스키마 불일치는 422가 될 수 있으므로 422만 보고 타임아웃으로 판단하지 않습니다.

서버 실행 후 Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## 7. 설치 및 실행

Python 3.12와 uv가 필요합니다. 저장소 루트에서 실행합니다.

```powershell
uv sync --locked
Copy-Item .env.example .env
```

이미 `.env`가 있으면 복사 명령을 생략하고 필요한 값만 수정합니다.
[.env.example](.env.example)은 기본 Solar 설정을 제공합니다. 실제 키는 Git에 커밋하지 않습니다.

```env
STOREPILOT_EMBEDDING_PROVIDER=solar
STOREPILOT_SOLAR_EMBEDDING_API_KEY=발급받은_키
STOREPILOT_SOLAR_EMBEDDING_QUERY_MODEL=solar-embedding-2-query
STOREPILOT_SOLAR_EMBEDDING_PASSAGE_MODEL=solar-embedding-2-passage
STOREPILOT_SOLAR_EMBEDDING_DIMENSIONS=1024
STOREPILOT_SOLAR_EMBEDDING_BATCH_SIZE=100
STOREPILOT_SOLAR_EMBEDDING_REQUESTS_PER_MINUTE=90

STOREPILOT_LLM_API_KEY=발급받은_키
STOREPILOT_LLM_MODEL=gpt-4o-mini
STOREPILOT_LLM_BASE_URL=https://api.openai.com/v1
STOREPILOT_LLM_TIMEOUT_SECONDS=90
STOREPILOT_LLM_BATCH_SIZE=15
STOREPILOT_LLM_MAX_CONCURRENCY=20
```

LLM 키는 선택 사항이지만, 미설정 시 애매한 결과는 첫 후보로 대체됩니다.
`app.main`이 프로젝트의 `.env`를 로드하며, 이미 설정된 프로세스 환경변수가 우선합니다. 설정 변경 후 서버를 재시작합니다.

```powershell
uv run python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

BE의 `AI_SERVER_BASE_URL`을 `http://127.0.0.1:8000`으로 설정합니다.
다른 서버에 배치한다면 사설 통신 경로와 접근 제한을 별도로 구성해야 합니다.

Windows에서 uv 경로를 별도로 지정해야 하는 경우:

```powershell
$env:UV_CACHE_DIR="C:\Project\StorePilot\ai-server\.uv-cache"
$env:UV_PYTHON_INSTALL_DIR="C:\Project\StorePilot\ai-server\.uv-python"
uv sync --locked
```

이 설정은 캐시·Python 설치 위치 지정일 뿐, Windows 애플리케이션 제어 정책의 실행 차단을 해제하지 않습니다.

## 8. 임베딩 설정

설정의 기준은 [settings.py](app/category_matcher/config/settings.py)입니다.

| Provider | 모델 기본값 | 역할 |
| --- | --- | --- |
| `solar` (기본) | `solar-embedding-2-query` / `solar-embedding-2-passage` | query/passage 분리 |
| `openai` | `text-embedding-3-small` | 동일 모델로 두 역할 처리 |
| `local` | `BAAI/bge-m3` | 로컬 sentence-transformers |

Solar의 기본 URL은 `https://api.upstage.ai/v1`, 호출 타임아웃은 60초입니다.
배치 크기는 코드에서 1~100으로 제한하고, dimensions는 응답 차원 검증에 사용합니다.

`REQUESTS_PER_MINUTE`는 **같은 프로세스의 모든 스레드에서 요청 시작 간격을 공유**합니다.
90이면 약 0.67초 간격이며, 0은 제한 해제입니다. 계정의 실제 한도에 맞게 조정해야 합니다.
토큰 한도·다른 프로세스/서비스의 호출량은 통제하지 않으며, **429 자동 재시도는 현재 없습니다.**

OpenAI 호환 API로 변경:

```env
STOREPILOT_EMBEDDING_PROVIDER=openai
STOREPILOT_EMBEDDING_API_KEY=발급받은_키
STOREPILOT_EMBEDDING_API_BASE_URL=https://api.openai.com/v1
STOREPILOT_EMBEDDING_API_MODEL=text-embedding-3-small
STOREPILOT_EMBEDDING_API_DIMENSIONS=1536
STOREPILOT_EMBEDDING_API_TIMEOUT_SECONDS=60
STOREPILOT_EMBEDDING_API_BATCH_SIZE=512
```

로컬 모델로 변경:

```env
STOREPILOT_EMBEDDING_PROVIDER=local
STOREPILOT_EMBEDDING_MODEL=BAAI/bge-m3
STOREPILOT_EMBEDDING_DEVICE=cpu
STOREPILOT_EMBEDDING_BATCH_SIZE=32
STOREPILOT_EMBEDDING_USE_FP16=false
```

코드의 장치 기본값은 `auto`이며 CUDA가 가능하면 CUDA, 아니면 CPU를 선택합니다. 위 예시와 `.env.example`은 CPU를 명시합니다.
로컬 모델은 첫 임베딩 요청에서 로드되고 이후 재사용됩니다. FP16은 CUDA에서만 적용하며, 별도의 PyTorch CPU 스레드 수 제한은 현재 없습니다.

## 9. 캐시와 인덱스 수명주기

```text
ai-cache/
├─ categories/<cache-key>/version-<versionId>/
│  ├─ category_embeddings.npy
│  ├─ category_meta.json
│  └─ model.json
└─ products/<cache-key>/shared/
   ├─ products.faiss
   └─ products.json
```

- 루트 경로: `STOREPILOT_AI_CACHE_ROOT`, `STOREPILOT_PRODUCT_CACHE_ROOT`
- Solar/OpenAI 캐시 키: provider·모델·차원 기반
- 로컬 캐시 키: 정규화한 모델명 기반
- 카테고리 캐시는 검색 시 디스크에서 읽고, 상품 인덱스는 첫 로드 후 프로세스 메모리에 유지합니다.
- 상품 인덱스 기본값은 `STOREPILOT_PRODUCT_INDEX_TYPE=flat` (`IndexFlatIP`, 정확 검색)입니다.
- `hnsw`도 지원합니다. 기본값은 M=32, efConstruction=200, efSearch=64이며 정확도·속도·메모리 비교 후 선택합니다.

모델/provider/차원을 변경하면 서버를 재시작하고 **카테고리 캐시와 상품 인덱스를 모두 새 설정으로 생성**합니다.
이전 설정의 유효한 캐시가 이미 있으면 재사용할 수 있지만, 데이터 최신성도 확인해야 합니다.
상품 인덱스 타입을 바꿀 때도 재생성이 필요합니다. 로컬 모델의 내용만 같은 이름 아래 변경하면 캐시 경로가 자동으로 달라지지 않습니다.

초기 데이터 준비 순서는 BE에서 네이버 카테고리 버전 업로드·임베딩 생성 → 기존 상품 및 매핑 파일로 상품 인덱스 생성 → 예측 확인입니다.
카테고리 캐시가 없거나 벡터 차원이 다르면 직접 검색 후보가 비고, 상품 인덱스가 없거나 호환되지 않으면 상품 검색 결과가 빕니다. `/health`는 이 상태를 검증하지 않습니다.

CLI로도 상품 인덱스를 재생성할 수 있습니다. 실행 중인 서버와 동시에 같은 캐시에 쓰지 말고, 완료 후 서버를 재시작해 메모리 인덱스를 갱신합니다.

```powershell
uv run python -m scripts.rebuild_product_index --mapping-json mappings.json products.xlsx
```

`mappings.json`은 API의 `categoryMappings`와 같은 객체 배열입니다. 원본 상품·매핑 파일은 재구축을 위해 별도로 보관합니다.

## 10. 동시 처리와 운영 제약

- API는 동기 함수이며 요청이 끝날 때까지 처리합니다. 영속 작업 큐·작업 상태 API는 AI 서버에 없고, 사용자 대상 비동기 작업은 BE가 관리합니다.
- LLM 동시성 20은 **예측 호출별 제한**입니다. 여러 요청의 합계나 여러 프로세스의 호출량을 제한하지 않습니다.
- 상품 변경에는 프로세스 내부 `RLock`이 있지만 검색 전체를 잠그지는 않습니다. 재생성의 임베딩 단계도 잠금 밖에 있으므로 재생성과 피드백을 동시에 수행하지 않도록 운영해야 합니다.
- 상품 파일은 임시 파일 작성 후 각각 교체하지만, FAISS와 JSON 두 파일이 하나의 트랜잭션으로 교체되는 것은 아닙니다. 카테고리 캐시 역시 여러 파일의 원자적 교체를 보장하지 않습니다.
- 여러 Uvicorn worker/서버 사이에 메모리 인덱스·락·Solar 속도 제한이 공유되지 않습니다. 단순히 worker를 늘리는 것만으로 안전한 확장이 되지 않습니다.
- 공유 상품 인덱스는 카테고리 버전별로 분리되지 않습니다. 네이버 카테고리 버전 변경 시 상품 라벨의 정합성을 확인해야 합니다.
- AI 라우트 자체에 인증·사용자별 할당량 검사가 없습니다. BE를 통해서만 접근하도록 네트워크를 제한해야 합니다.
- 현재 Prometheus/Grafana 계측이나 GPU 전용 검색은 구현되어 있지 않습니다. 서버 사양 및 처리 성능은 데이터량·동시 요청·API 한도에 맞춰 측정해야 합니다.

## 11. 테스트와 성능 확인

```powershell
uv run python -m unittest discover tests
uv run python -m compileall app tests
```

테스트는 전처리, 카테고리 벡터 검색, 상품 인덱스 재생성/피드백, 충돌 근거 제외, 자동 선택, 대칭/비대칭 분기, Alias, LLM 배치 및 Solar 요청 간격 등을 검증합니다.
외부 API를 대체한 단위 테스트이므로 실제 API 연결·추천 정확도·운영 부하 검증을 대신하지 않습니다.

| 로그 | 확인할 내용 |
| --- | --- |
| `category_predict_timing` | 전체 시간, 상품/카테고리 임베딩, FAISS, 직접 검색, LLM 단계 시간 |
| `embedding_api_timing` | provider·모델·query/passage·입력 수·토큰 사용량·응답시간 |
| `embedding_api_rate_limit` | Solar 요청 간격 대기 |
| `llm_category_batch_timing` | LLM 상품 수·배치 수·동시성·전체 시간 |
| `llm_category_chunk_result` | 선택·거절·실패 수 |
| `Product embedding progress` | 상품 인덱스 재생성 진행량 |

Solar의 `embedding_api_timing`은 속도 제한 대기 이후부터 측정합니다. 예측의 `embedding_ms`에는 해당 대기 시간이 포함됩니다.
BE의 k6 테스트와 함께 동일 파일·모델·인덱스·동시 사용자 수를 고정해 비교하고, 응답시간 외에 정답 라벨 대비 정확도와 LLM/API 비용도 별도로 확인합니다.
