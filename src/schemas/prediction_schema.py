from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="SMS text to classify"
    )


class PredictionResponse(BaseModel):
    text: str
    prediction: str


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool