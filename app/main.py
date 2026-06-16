from fastapi import FastAPI

from app.category_matcher.router import router as category_matcher_router

app = FastAPI(title="StorePilot AI Server")
app.include_router(category_matcher_router, prefix="/ai/categories", tags=["Category Matcher"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
