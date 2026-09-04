app = FastAPI()


class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    intent: str
    confidence: float


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):
    result = predict_intent(
        message=request.text
    )

    return result