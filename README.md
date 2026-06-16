# StorePilot AI Server

상품명으로 네이버 카테고리를 찾기 위한 FastAPI 서버입니다.

## Python Version

`sentence-transformers`는 내부적으로 PyTorch를 사용합니다. PyTorch 계열 패키지는 최신 Python 버전 지원이 늦게 붙는 경우가 많아서, 이 서버는 uv로 Python 3.12를 고정해서 실행합니다.

```text
requires-python = ">=3.12,<3.13"
```

## Setup With uv

```bash
cd ai-server
uv python install 3.12
uv sync
```

## Run

```bash
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Flow

1. Spring Boot uploads Naver categories.
2. Spring Boot calls `POST /ai/categories/rebuild`.
3. The AI server embeds categories with `intfloat/multilingual-e5-small`.
4. Product names are sent to `POST /ai/categories/predict`.
5. The response returns Top 1 Naver category per product.
