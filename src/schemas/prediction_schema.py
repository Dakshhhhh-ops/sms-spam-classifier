from pydantic import BaseModel


class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    text: str
    prediction: str
    spam_probability: float
    ham_probability: float


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool