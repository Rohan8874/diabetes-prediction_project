import json
import os
from typing import Dict

import anyio
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .schemas import PatientInput, PredictionResponse

MODEL_PATH = os.path.join("model", "diabetes_model.pkl")
METRICS_PATH = os.path.join("metrics", "metrics.json")

app = FastAPI(title="Diabetes Prediction API", version="1.0.0")

# Allow all origins for demo; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
meta = bundle["meta"]
FEATURES = meta["feature_order"]

with open(METRICS_PATH, "r") as f:
    METRICS = json.load(f)

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.get("/metrics")
async def metrics():
    return METRICS

@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: PatientInput):
    # Ensure feature order is consistent with training
    row = pd.DataFrame([[getattr(payload, f) for f in FEATURES]], columns=FEATURES)

    def _infer():
        proba = None
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(row)
            pred = int(np.argmax(proba[0]))
            confidence = float(np.max(proba[0]))
        else:
            # fallback (e.g., if probability not available)
            # normalize decision function to pseudo-confidence
            pred = int(pipeline.predict(row)[0])
            confidence = 0.5
        return pred, confidence

    pred, confidence = await anyio.to_thread.run_sync(_infer)
    result = "Diabetic" if pred == 1 else "Not Diabetic"
    return PredictionResponse(prediction=pred, result=result, confidence=round(confidence, 4))
