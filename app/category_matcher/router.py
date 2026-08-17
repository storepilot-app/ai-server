import json
import zipfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.category_matcher.product_memory.store import (
    NaverCategoryLabel,
    add_product_feedback,
    add_product_feedbacks,
    rebuild_product_index,
)
from app.category_matcher.schemas import (
    MyCategoryMappingItem,
    PredictRequest,
    PredictResponse,
    ProductFeedbackBatchRequest,
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
        )
    )


@router.post("/product-index/rebuild", response_model=ProductIndexRebuildResponse)
def rebuild_products(
    user_id: int = Form(alias="userId"),
    category_mappings: str = Form(alias="categoryMappings"),
    files: list[UploadFile] = File(),
) -> ProductIndexRebuildResponse:
    if user_id <= 0:
        raise HTTPException(status_code=400, detail="userId is required.")
    if not files:
        raise HTTPException(status_code=400, detail="At least one Excel file is required.")
    if any(not (file.filename or "").lower().endswith(".xlsx") for file in files):
        raise HTTPException(status_code=400, detail="Only .xlsx files are supported.")

    try:
        mapping_items = [
            MyCategoryMappingItem.model_validate(item)
            for item in json.loads(category_mappings)
        ]
        mappings = {
            item.myCategoryCode: NaverCategoryLabel(
                category_id=item.categoryId,
                category_code=item.categoryCode,
                full_path=item.fullPath,
            )
            for item in mapping_items
        }
        result = rebuild_product_index([file.file for file in files], mappings)
    except (json.JSONDecodeError, ValueError, OSError, zipfile.BadZipFile) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    return ProductIndexRebuildResponse(
        userId=user_id,
        sourceCount=len(files),
        sourceRowCount=result.source_row_count,
        validRowCount=result.valid_row_count,
        unmappedRowCount=result.unmapped_row_count,
        indexedProductCount=result.indexed_product_count,
        duplicateRowCount=result.duplicate_row_count,
        conflictingTitleCount=result.conflicting_title_count,
        message="Historical product index rebuilt.",
    )


@router.post("/product-index/feedback", response_model=ProductFeedbackResponse)
def add_feedback(request: ProductFeedbackRequest) -> ProductFeedbackResponse:
    try:
        count = add_product_feedback(
            request.productName,
            NaverCategoryLabel(
                category_id=request.categoryId,
                category_code=request.categoryCode,
                full_path=request.fullPath,
            ),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return ProductFeedbackResponse(
        userId=request.userId,
        indexedProductCount=count,
        message="Product correction added to the search index.",
    )


@router.post("/product-index/feedback/batch", response_model=ProductFeedbackResponse)
def add_feedbacks(request: ProductFeedbackBatchRequest) -> ProductFeedbackResponse:
    try:
        count = add_product_feedbacks([
            (
                product.productName,
                NaverCategoryLabel(
                    category_id=product.categoryId,
                    category_code=product.categoryCode,
                    full_path=product.fullPath,
                ),
            )
            for product in request.products
        ])
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return ProductFeedbackResponse(
        userId=request.userId,
        indexedProductCount=count,
        message="Product corrections added to the search index.",
    )
