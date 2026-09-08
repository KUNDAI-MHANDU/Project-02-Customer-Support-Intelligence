from typing import Annotated
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel, StringConstraints
from fastapi import FastAPI, HTTPException

from src.predict import predict_intent
from src.database.crud import save_ticket, get_ticket_by_id

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

class TicketResponse(BaseModel):
    ticket_id: int
    text: str
    predicted_intent: str
    confidence: float
    created_at: datetime

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

# Define the /tickets/{ticket_id} endpoint, which takes a ticket ID and returns a TicketResponse
@app.get(
    "/tickets/{ticket_id}",
    response_model=TicketResponse
)
def get_ticket(ticket_id: int):

    # Get the ticket with the specified ID from the database using the get_ticket_by_id function from src.database.crud
    ticket = get_ticket_by_id(ticket_id)

    # If the ticket is not found, raise an HTTPException with a 404 status code and a "Ticket not found" detail
    if ticket is None:
        raise HTTPException(
            status_code=404,
            detail="Ticket not found"
        )

    # Return the ticket ID, text, predicted intent, confidence score, and creation timestamp in the response as a dictionary
    return {
        "ticket_id": ticket.id,
        "text": ticket.text,
        "predicted_intent": ticket.predicted_intent,
        "confidence": ticket.confidence,
        "created_at": ticket.created_at
    }