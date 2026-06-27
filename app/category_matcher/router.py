import zipfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.category_matcher.product_memory.store import add_product_feedback, rebuild_product_index
from app.category_matcher.schemas import (
    PredictRequest,
    PredictResponse,
    ProductFeedbackRequest,
    ProductFeedbackResponse,
    ProductIndexRebuildResponse,
    RebuildRequest,
    RebuildResponse,
)
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
    return PredictResponse(
        results=predict_categories(
            request.versionId,
            request.products,
            request.userKey,
            request.myCategoryMappings,
        )
    )


@router.post("/product-index/rebuild", response_model=ProductIndexRebuildResponse)
def rebuild_products(
    user_key: str = Form(alias="userKey"),
    files: list[UploadFile] = File(),
) -> ProductIndexRebuildResponse:
    if not user_key.strip():
        raise HTTPException(status_code=400, detail="userKey is required.")
    if not files:
        raise HTTPException(status_code=400, detail="At least one Excel file is required.")
    if any(not (file.filename or "").lower().endswith(".xlsx") for file in files):
        raise HTTPException(status_code=400, detail="Only .xlsx files are supported.")

    try:
        result = rebuild_product_index(user_key, [file.file for file in files])
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return ProductIndexRebuildResponse(
        userKey=user_key.strip(),
        sourceCount=len(files),
        validRowCount=result.valid_row_count,
        indexedProductCount=result.indexed_product_count,
        duplicateRowCount=result.duplicate_row_count,
        conflictingTitleCount=result.conflicting_title_count,
        message="Historical product index rebuilt.",
    )


@router.post("/product-index/feedback", response_model=ProductFeedbackResponse)
def add_feedback(request: ProductFeedbackRequest) -> ProductFeedbackResponse:
    try:
        count = add_product_feedback(request.userKey, request.productName, request.myCategoryCode)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return ProductFeedbackResponse(
        userKey=request.userKey.strip(),
        indexedProductCount=count,
        message="Product correction added to the search index.",
    )
