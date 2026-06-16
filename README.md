# StorePilot AI Server

상품명으로 네이버 카테고리를 찾기 위한 FastAPI 서버입니다.

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Flow

1. Spring Boot uploads Naver categories.
2. Spring Boot calls `POST /ai/categories/rebuild`.
3. The AI server embeds categories with `intfloat/multilingual-e5-small`.
4. Product names are sent to `POST /ai/categories/predict`.
5. The response returns Top 1 Naver category per product.
