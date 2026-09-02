import joblib as jb
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np 
model = jb.load("/app/model/svm_classification.joblib")


app = FastAPI()


@app.get("/health")
async def root():
    return {"status": "ok"}
class PredictionRequest(BaseModel):
    features: list[float]


@app.post("/api/predict")
async def predict(request: PredictionRequest):
    features = request.features
    X = np.array([features])

   

    prediction = model.predict(X)
    probability = model.predict_proba(X)

    return {
        "prediction": int(prediction[0]),
        "probability": float(probability[0].max())
    }