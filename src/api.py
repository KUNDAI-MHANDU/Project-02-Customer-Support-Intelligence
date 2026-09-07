from typing import Annotated

from fastapi import FastAPI
from pydantic import BaseModel, StringConstraints

from src.predict import predict_intent
from src.database.crud import save_ticket

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
    ticket_id: int
    intent: str
    confidence: float

# Define the /predict endpoint, which takes a PredictionRequest and returns a PredictionResponse
@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):
    result = predict_intent(
        message=request.text
    )

    # Save the ticket to the database using the save_ticket function from src.database.crud
    ticket = save_ticket(
        text=request.text,
        predicted_intent=result["intent"],
        confidence=result["confidence"]
    )

    # Return the ticket ID, predicted intent, and confidence score in the response as a dictionary
    return {
        "ticket_id": ticket.id,
        "intent": result["intent"],
        "confidence": result["confidence"]
    }