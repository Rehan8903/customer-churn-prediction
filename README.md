# Churn Signal — Customer Churn Prediction (End-to-End MLOps)

An end-to-end machine learning pipeline that predicts customer churn risk for a telecom provider — from raw data to a live, containerized prediction app, with full experiment tracking, CI/CD, and a model registry gating what actually reaches production.

---

## Overview

Customer churn prediction on the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), built as a fully reproducible ML pipeline rather than a one-off notebook. The project covers the full lifecycle: data ingestion, preprocessing, feature engineering, model training, evaluation, registration, automated testing, and deployment via Docker.

## Tech stack

| Layer | Tools |
|---|---|
| Pipeline orchestration | DVC |
| Experiment tracking & model registry | MLflow (hosted on DagsHub) |
| Model | Logistic Regression (scikit-learn) |
| Backend / API | FastAPI |
| Frontend | HTML / CSS / vanilla JS |
| CI/CD | GitHub Actions |
| Containerization | Docker |
| Data versioning | DVC + DagsHub remote storage |

## Architecture

```
Raw Data (Telco CSV)
      │
      ▼
 Data Ingestion ──▶ Preprocessing ──▶ Feature Engineering ──▶ Model Training
                                                                     │
                                                                     ▼
                                                            Model Evaluation
                                                                     │
                                                                     ▼
                                                          Model Registration (MLflow)
                                                                     │
                                                                     ▼
                                                    load_model_test.py (CI quality gate)
                                                                     │
                                                                     ▼
                                                       FastAPI app (Docker container)
                                                                     │
                                                                     ▼
                                                              Web UI / predictions
```

Every stage is a DVC pipeline stage (`dvc.yaml`), so the entire chain reproduces with a single command, and only re-runs the stages actually affected by a given change (new data, new hyperparameters, new code).

## Project structure

```
customer-churn-prediction/
├── .github/workflows/       # CI/CD pipeline (GitHub Actions)
├── data/                    # DVC-tracked data (raw, interim, processed) — gitignored
├── fastapi_app/
│   ├── templates/           # index.html, style.css, script.js
│   ├── app.py                # FastAPI app + /predict endpoint
│   ├── preprocessing_utility.py  # shared train/serve preprocessing logic
│   └── load_model_test.py    # CI model-quality gate (pytest)
├── models/                   # trained model, encoders, scaler — DVC-tracked
├── src/
│   ├── data/                 # ingestion, preprocessing scripts
│   ├── features/              # feature engineering script
│   ├── model/                 # training, evaluation, registration scripts
│   └── logger/                 # shared logging config
├── reports/                    # metrics.json, experiment_info.json
├── dvc.yaml                    # pipeline stage definitions
├── params.yaml                  # tracked hyperparameters
├── requirements.txt
├── Dockerfile
└── README.md
```

## Dataset

- **Source:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle)
- ~7,000 customer records, 20 features (demographics, account info, services subscribed)
- Target: binary churn (Yes/No), ~26% positive class — handled with `class_weight='balanced'`


## Running locally

**1. Clone and set up the environment**
```bash
git clone https://github.com/Rehan8903/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
```

**2. Set your DagsHub token** (for MLflow tracking)
```bash
export CHURN_PREDICTION_TOKEN=your_dagshub_token   # Windows: set / $env:
```

**3. Reproduce the full pipeline**
```bash
dvc repro
```

**4. Run the app**
```bash
cd fastapi_app
uvicorn app:app --reload
```
Visit `http://localhost:8000`

## Running with Docker

```bash
docker build -t churn-signal:latest .
docker run -p 8000:8000 -e CHURN_PREDICTION_TOKEN=your_dagshub_token churn-signal:latest
```

## CI/CD

Every push to `main` triggers a GitHub Actions workflow that:
1. Reproduces the full DVC pipeline
2. Runs `load_model_test.py` — a quality gate that blocks a model from being trusted if it doesn't clear minimum accuracy/F1/recall thresholds against the held-out test set
3. Pushes pipeline outputs to DVC remote storage

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## API

**POST `/predict`**

Request body:
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
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
  "MonthlyCharges": 85.50,
  "TotalCharges": 1026.00
}
```

Response:
```json
{ "churn_probability": 0.73 }
```

## Future improvements

- Promote model from `staging` to `production` alias with an approval step
- Add data drift monitoring on incoming prediction requests
- Push Docker image to a registry (Docker Hub / AWS ECR) for actual deployment
- Add integration tests for the `/predict` endpoint itself, not just the model

## Author

**Rehan Sarfraz**
https://lnkd.in/p/d5T_auTK
