from fastapi import FastAPI, HTTPException, status

from src.schemas.prediction_schema import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse
)

from src.services.prediction_service import PredictionService

from src.exception import CustomException

app = FastAPI(
    title="SMS Spam Classifier API",
    description="""
    API for classifying SMS messages into:

    - Spam
    - Ham

    Built using:
    - FastAPI
    - Scikit-Learn
    - MLflow
    """,
    version="1.0.0",
)

service = PredictionService()


@app.get(
    "/",
    tags=["Home"]
)
def home():

    return {
        "message": "SMS Spam Classifier API is running"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"]
)
def health():

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=True
    )


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"]
)
def predict(data: PredictionRequest):

    try:

        prediction = service.predict(
            data.text
        )

        result = service.predict(
            data.text
        )
        print("RESULT:", result)

        return {
            "text": data.text,
            "prediction": result["prediction"],
            "spam_probability": result["spam_probability"],
            "ham_probability": result["ham_probability"]
        }

    except CustomException as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Unexpected Error: {str(e)}"
        )