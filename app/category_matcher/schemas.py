from pydantic import BaseModel, Field


class CategoryItem(BaseModel):
    categoryId: int
    categoryCode: str
    fullPath: str
    searchText: str


class RebuildRequest(BaseModel):
    versionId: int
    categories: list[CategoryItem]


class RebuildResponse(BaseModel):
    versionId: int
    categoryCount: int
    message: str


class ProductItem(BaseModel):
    rowId: int
    productName: str


class PredictRequest(BaseModel):
    versionId: int
    products: list[ProductItem] = Field(default_factory=list)


class PredictionItem(BaseModel):
    rowId: int
    categoryId: int | None
    categoryCode: str | None
    fullPath: str | None
    score: float


class PredictResponse(BaseModel):
    results: list[PredictionItem]
