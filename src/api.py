from typing import Annotated

from fastapi import FastAPI
from pydantic import BaseModel, StringConstraints

from src.predict import predict_intent

# Create a FastAPI app and define the request and response models for the /predict endpoint
app = FastAPI()

# Define the request and response models for the /predict endpoint using Pydantic's BaseModel and StringConstraints
class PredictionRequest(BaseModel):
    text: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=1000
        )
    ]


class PredictionResponse(BaseModel):
    intent: str
    confidence: float

# Define the /predict endpoint for the FastAPI app, which takes a PredictionRequest and returns a PredictionResponse
@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):
    result = predict_intent(
        message=request.text
    )

    return result