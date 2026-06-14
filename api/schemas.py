from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10_000,
                      description="Full text of the news article")

class PredictResponse(BaseModel):
    label: str                  # "REAL" or "FAKE"
    confidence: float           # 0.00 – 100.00
    fake_probability: float
    real_probability: float
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

class ModelInfoResponse(BaseModel):
    model_type: str
    vocabulary_size: int
    n_features: int

class VerifyArticleResponse(BaseModel):
    title: str
    source: str
    published_at: str
    url: str
    relevance_score: float

class VerifyResponse(BaseModel):
    articles: list[VerifyArticleResponse]

