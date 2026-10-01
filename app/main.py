from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.utils import predict

app = FastAPI()


class PredictionRequest(BaseModel):
    features: list[float] = Field(..., min_length=1)


@app.post("/predict")
def predict_endpoint(data: PredictionRequest):
    if not data.features:
        raise HTTPException(status_code=422, detail="features must not be empty")
    predictions = predict(data.features)
    return {"predictions": predictions}


@app.get("/")
def read_root():
    return {"message": "API is up and running!"}


@app.get("/favicon.ico")
def favicon():
    raise HTTPException(status_code=404, detail="Not Found")
