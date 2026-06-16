from fastapi import APIRouter

from app.category_matcher.schemas import PredictRequest, PredictResponse, RebuildRequest, RebuildResponse
from app.category_matcher.service import predict_categories, rebuild_category_cache

router = APIRouter()


@router.post("/rebuild", response_model=RebuildResponse)
def rebuild(request: RebuildRequest) -> RebuildResponse:
    rebuild_category_cache(request.versionId, request.categories)
    return RebuildResponse(
        versionId=request.versionId,
        categoryCount=len(request.categories),
        message="Category embeddings rebuilt.",
    )


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    return PredictResponse(results=predict_categories(request.versionId, request.products))
