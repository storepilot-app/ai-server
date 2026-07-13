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


class MyCategoryMappingItem(BaseModel):
    myCategoryCode: str
    categoryId: int
    categoryCode: str
    fullPath: str


class PredictRequest(BaseModel):
    versionId: int
    products: list[ProductItem] = Field(default_factory=list)


class PredictionCandidate(BaseModel):
    categoryId: int
    categoryCode: str
    fullPath: str
    score: float


class SimilarProductItem(BaseModel):
    productName: str
    categoryId: int
    categoryCode: str
    fullPath: str
    similarity: float


class CategoryDistributionItem(BaseModel):
    categoryId: int
    categoryCode: str
    fullPath: str
    support: float
    exampleCount: int
    maxSimilarity: float


class PredictionItem(BaseModel):
    rowId: int
    categoryId: int | None
    categoryCode: str | None
    fullPath: str | None
    score: float
    candidates: list[PredictionCandidate] = Field(default_factory=list)
    llmUsed: bool = False
    llmSelectedCategory: str | None = None
    llmStatus: str = "SKIPPED"
    llmStatusDetail: str | None = None
    decisionSource: str = "CATEGORY_EMBEDDING"
    similarProducts: list[SimilarProductItem] = Field(default_factory=list)
    categoryDistribution: list[CategoryDistributionItem] = Field(default_factory=list)


class PredictResponse(BaseModel):
    results: list[PredictionItem]


class ProductIndexRebuildResponse(BaseModel):
    userId: int
    sourceCount: int
    sourceRowCount: int
    validRowCount: int
    unmappedRowCount: int
    indexedProductCount: int
    duplicateRowCount: int
    conflictingTitleCount: int
    message: str


class ProductFeedbackRequest(BaseModel):
    userId: int
    productName: str
    categoryId: int
    categoryCode: str
    fullPath: str


class ProductFeedbackResponse(BaseModel):
    userId: int
    indexedProductCount: int
    message: str
