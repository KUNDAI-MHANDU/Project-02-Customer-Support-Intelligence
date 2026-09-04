from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_intent


# Create a FastAPI app and define a POST endpoint for predicting the intent of a given message using the trained classifier and the label map
app = FastAPI()

# Define a Pydantic model for the request body of the POST endpoint, which contains a single field "text" of type string
class PredictionRequest(BaseModel):
    text: str

# Define a POST endpoint for predicting the intent of a given message using the trained classifier and the label map
@app.post("/predict")
def predict(request: PredictionRequest):

    result = predict_intent(
        message=request.text
    )

    return result