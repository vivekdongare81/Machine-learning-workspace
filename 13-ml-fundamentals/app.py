from pathlib import Path
import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = Path("artifacts/breast_cancer_pipeline.joblib")
MODEL_VERSION = "1.0.0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml-serving")
app = FastAPI(title="ML Prediction API", version=MODEL_VERSION)

# Load once when the process starts, not on every request.
# If loading fails, keep the process up but report "not ready" (503) instead of crashing.
try:
    model = joblib.load(MODEL_PATH)
    expected_features = list(model.feature_names_in_)
except Exception:
    logger.exception("Model failed to load")
    model, expected_features = None, []

class PredictionRequest(BaseModel):
    features: dict[str, float]

class PredictionResponse(BaseModel):
    prediction: int
    probability_class_1: float
    model_version: str

@app.get("/health")
def health():
    # Liveness: the process can respond.
    return {"status": "ok"}

@app.get("/ready")
def ready():
    # Readiness: model is loaded and the service can serve.
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ready", "model_version": MODEL_VERSION}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    missing = set(expected_features) - set(request.features)
    extra = set(request.features) - set(expected_features)
    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": sorted(missing),
                    "extra_features": sorted(extra)}
        )
    try:
        row = pd.DataFrame(
            [[request.features[name] for name in expected_features]],
            columns=expected_features
        )
        return PredictionResponse(
            prediction=int(model.predict(row)[0]),
            probability_class_1=float(model.predict_proba(row)[0, 1]),
            model_version=MODEL_VERSION
        )
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")
