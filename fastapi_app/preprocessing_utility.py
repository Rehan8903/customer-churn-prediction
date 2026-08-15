# preprocessing_utility.py
"""
Shared preprocessing logic for the churn prediction pipeline.

This module applies the exact same transformation used during training
(label encoding with the fitted encoders, then StandardScaler) to raw
inference-time input, so predictions don't suffer from train/serve skew.

Used by:
- fastapi_app/app.py               (inference time)
- src/data/data_preprocessing.py   (optionally, if you refactor training to
  import from here instead of duplicating the encoding logic)
"""

import pickle
from pathlib import Path

import pandas as pd

from src.logger import logging

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / "models"  # adjust if models/ lives elsewhere relative to this file

ENCODERS_PATH = MODELS_DIR / "label_encoders.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

_label_encoders = None
_scaler = None


def _load_artifacts():
    """Lazily load the fitted encoders and scaler once, then cache them."""
    global _label_encoders, _scaler

    if _label_encoders is None:
        try:
            with open(ENCODERS_PATH, "rb") as f:
                _label_encoders = pickle.load(f)
            logging.info("Loaded label encoders from %s", ENCODERS_PATH)
        except FileNotFoundError:
            logging.error("label_encoders.pkl not found at %s", ENCODERS_PATH)
            raise

    if _scaler is None:
        try:
            with open(SCALER_PATH, "rb") as f:
                _scaler = pickle.load(f)
            logging.info("Loaded scaler from %s", SCALER_PATH)
        except FileNotFoundError:
            logging.error("scaler.pkl not found at %s", SCALER_PATH)
            raise

    return _label_encoders, _scaler


def encode_categoricals(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """
    Apply the saved LabelEncoders to every categorical column they cover.
    Unseen categories (values that never appeared in training) map to -1
    instead of raising, matching the fallback used in data_preprocessing.py.
    """
    df = df.copy()
    for col, le in encoders.items():
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).map(
            lambda val, le=le: le.transform([val])[0] if val in le.classes_ else -1
        )
    return df


def preprocess_input(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw customer data into the exact numeric, scaled format the
    trained model expects.

    Args:
        raw_df: single-row (or multi-row) DataFrame with the raw Telco-style
                columns collected from the form / API request.

    Returns:
        pd.DataFrame: scaled features, column order aligned to what the
        model was trained on, ready to pass into model.predict_proba().
    """
    try:
        encoders, scaler = _load_artifacts()

        df = raw_df.copy()

        # Encode categorical columns using the fitted training-time encoders
        df = encode_categoricals(df, encoders)

        # Align column order to exactly what the scaler was fit on.
        # StandardScaler stores this automatically when fit on a DataFrame
        # (scikit-learn >= 1.0), via scaler.feature_names_in_ — this avoids
        # hardcoding a column list that could silently drift out of sync
        # with your actual training pipeline.
        if hasattr(scaler, "feature_names_in_"):
            expected_cols = list(scaler.feature_names_in_)
            missing = [c for c in expected_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing expected columns for scaling: {missing}")
            df = df[expected_cols]
        else:
            logging.warning(
                "Scaler has no feature_names_in_ — using incoming column order as-is. "
                "Verify this matches training-time column order exactly."
            )

        scaled_array = scaler.transform(df)
        scaled_df = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)

        logging.info("Preprocessed %d row(s) for inference", len(scaled_df))
        return scaled_df

    except Exception as e:
        logging.error("Error during inference preprocessing: %s", e)
        raise