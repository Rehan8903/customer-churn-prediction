# load_model_test.py
"""
Sanity checks run before promoting a model version — typically wired into
CI (GitHub Actions) right after model_registration, and before anything
moves from the "staging" alias to "production".

Run with:
    pytest fastapi_app/load_model_test.py -v

What this checks:
1. The registered model actually loads from the MLflow registry.
2. It accepts a realistic single-row input and returns a valid probability.
3. It clears a minimum performance bar against the held-out test set —
   this stops a genuinely worse model from silently reaching production.
"""

import os
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, recall_score

from src.logger import logging
from preprocessing_utility import preprocess_input

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test_scaled.csv"

MODEL_NAME = "churn_prediction_model"
MODEL_ALIAS = "staging"
TARGET_COL = "Churn"

# Minimum bar a model must clear to be considered safe to promote.
# Set relative to your own experiment history — your Logistic Regression +
# Label Encoding run hit F1 ~0.617 / recall ~0.794, so these thresholds sit
# a bit below that as a "don't regress badly" floor, not a "must match best" bar.
MIN_ACCURACY = 0.70
MIN_F1 = 0.55
MIN_RECALL = 0.65


@pytest.fixture(scope="module")
def dagshub_mlflow_setup():
    dagshub_token = os.getenv("CHURN_PREDICTION_TOKEN")
    if not dagshub_token:
        pytest.skip("CHURN_PREDICTION_TOKEN not set — skipping registry tests")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    mlflow.set_tracking_uri(
        "https://dagshub.com/rehansarfraz8903/Customer-Churn-prediction.mlflow"
    )


@pytest.fixture(scope="module")
def loaded_model(dagshub_mlflow_setup):
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.pyfunc.load_model(model_uri)
    logging.info("Test fixture loaded model from %s", model_uri)
    return model


def _get_sklearn_model(pyfunc_model):
    """Unwrap the underlying sklearn estimator to access predict_proba()."""
    return pyfunc_model._model_impl.sklearn_model


# ---------------------------------------------------------------------------
# 1. Does the model load at all?
# ---------------------------------------------------------------------------
def test_model_loads_from_registry(loaded_model):
    assert loaded_model is not None


# ---------------------------------------------------------------------------
# 2. Does it handle a realistic single input and return a valid probability?
# ---------------------------------------------------------------------------
def test_single_prediction_is_valid_probability(loaded_model):
    sample = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.50,
        "TotalCharges": 477.50,
    }])

    processed = preprocess_input(sample)
    sk_model = _get_sklearn_model(loaded_model)
    probability = sk_model.predict_proba(processed)[:, 1][0]

    assert 0.0 <= probability <= 1.0
    logging.info("Sample prediction probability: %.4f", probability)


# ---------------------------------------------------------------------------
# 3. Does it clear the minimum performance bar on the real held-out test set?
# ---------------------------------------------------------------------------
def test_model_meets_minimum_performance(loaded_model):
    if not TEST_DATA_PATH.exists():
        pytest.skip(f"Test data not found at {TEST_DATA_PATH} — run the DVC pipeline first")

    test_data = pd.read_csv(TEST_DATA_PATH)
    X_test = test_data.drop(columns=[TARGET_COL])
    y_test = test_data[TARGET_COL]

    sk_model = _get_sklearn_model(loaded_model)
    y_pred = sk_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    logging.info(
        "Registered model performance — accuracy: %.4f, f1: %.4f, recall: %.4f",
        accuracy, f1, recall,
    )

    assert accuracy >= MIN_ACCURACY, f"Accuracy {accuracy:.4f} below floor {MIN_ACCURACY}"
    assert f1 >= MIN_F1, f"F1 {f1:.4f} below floor {MIN_F1}"
    assert recall >= MIN_RECALL, f"Recall {recall:.4f} below floor {MIN_RECALL}"