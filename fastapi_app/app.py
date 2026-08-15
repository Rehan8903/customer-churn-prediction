# app.py

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel

from src.logger import logging
from fastapi_app.preprocessing_utility import preprocess_input

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
MODELS_DIR = BASE_DIR.parent / "models"   # models/ sits at project root, one level up from fastapi_app/

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ---------------------------------------------------------------------------
# MLflow / DagsHub config (for pulling the registered model)
# ---------------------------------------------------------------------------
dagshub_token = os.getenv("CHURN_PREDICTION_TOKEN")
dagshub_url = "https://dagshub.com"
repo_owner = "rehansarfraz8903"
repo_name = "Customer-Churn-prediction"
MODEL_NAME = "churn_prediction_model"
MODEL_ALIAS = "staging"

model = None

def load_production_model():
    """
    Load the registered model from the MLflow registry (preferred, since it
    always reflects whatever version currently holds the 'staging' alias).
    Falls back to the local models/model.pkl if the registry is unreachable
    (no token, no network) so the app can still run locally.
    """
    global model
    try:
        if not dagshub_token:
            raise EnvironmentError("CHURN_PREDICTION_TOKEN not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        model = mlflow.pyfunc.load_model(model_uri)
        logging.info("Loaded model from MLflow registry: %s", model_uri)

    except Exception as e:
        logging.warning("Could not load model from MLflow registry (%s). Falling back to local models/model.pkl", e)
        local_model_path = MODELS_DIR / "model.pkl"
        with open(local_model_path, "rb") as f:
            model = pickle.load(f)
        logging.info("Loaded model from local file: %s", local_model_path)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Churn Signal", description="Customer churn risk prediction API")


@app.on_event("startup")
def startup_event():
    load_production_model()


# ---------------------------------------------------------------------------
# Request schema — matches the payload built in script.js's collectPayload()
# ---------------------------------------------------------------------------
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def serve_index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/style.css")
def serve_css():
    return FileResponse(TEMPLATES_DIR / "style.css", media_type="text/css")


@app.get("/script.js")
def serve_js():
    return FileResponse(TEMPLATES_DIR / "script.js", media_type="text/javascript")


@app.post("/predict")
def predict(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        raw_df = pd.DataFrame([customer.dict()])
        logging.info("Received prediction request: %s", customer.dict())

        processed_df = preprocess_input(raw_df)

        # mlflow.pyfunc models expose .predict(); local sklearn model exposes
        # both .predict() and .predict_proba() — handle whichever we loaded
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(processed_df)[:, 1][0])
        else:
            # pyfunc-wrapped sklearn classifiers return class predictions by
            # default; unwrap the underlying sklearn model to get probabilities
            sk_model = model._model_impl.sklearn_model if hasattr(model, "_model_impl") else model
            probability = float(sk_model.predict_proba(processed_df)[:, 1][0])

        logging.info("Prediction complete. Churn probability: %.4f", probability)
        return {"churn_probability": probability}

    except Exception as e:
        logging.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app.app:app", host="0.0.0.0", port=8000, reload=True)