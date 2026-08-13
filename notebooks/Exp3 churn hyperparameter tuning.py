import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# ==========================
# MLflow / DagsHub setup
# ==========================
MLFLOW_TRACKING_URI = "https://dagshub.com/rehansarfraz8903/Customer-Churn-prediction.mlflow"
dagshub.init(repo_owner="rehansarfraz8903", repo_name="Customer-Churn-prediction", mlflow=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("GradientBoosting Hyperparameter Tuning")


# ==========================
# Load & Prepare Data
# ==========================
def load_and_prepare_data(filepath):
    """Loads the churn dataset, cleans it, and one-hot encodes it (exp2's best-performing encoding)."""
    df = pd.read_csv(filepath)
    df.drop(columns=["customerID"], inplace=True, errors="ignore")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).infer_objects(copy=False)

    y = df["Churn"]
    X_raw = df.drop(columns=["Churn"])

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = X_raw.select_dtypes(include="object").columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
    ])
    X = preprocessor.fit_transform(X_raw)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ==========================
# Train & Log Model
# ==========================
def train_and_log_model(X_train, X_test, y_train, y_test):
    """Runs GridSearchCV over GradientBoosting hyperparameters and logs every combo to MLflow."""

    param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 4],
    }

    # scoring="f1" balances precision/recall; swap to "recall" if catching
    # churners matters more to you than false-positive rate right now
    scoring = "f1"

    with mlflow.start_run(run_name="GB Grid Search"):
        grid_search = GridSearchCV(
            GradientBoostingClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring=scoring,
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)

        # Log every hyperparameter combo as its own nested run
        for params, mean_score, std_score in zip(
            grid_search.cv_results_["params"],
            grid_search.cv_results_["mean_test_score"],
            grid_search.cv_results_["std_test_score"],
        ):
            with mlflow.start_run(run_name=f"GB with params: {params}", nested=True):
                model = GradientBoostingClassifier(random_state=42, **params)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred),
                    "mean_cv_score": mean_score,
                    "std_cv_score": std_score,
                }

                mlflow.log_params(params)
                mlflow.log_metrics(metrics)

                print(f"Params: {params} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1_score']:.4f}")

        # Log the best model from the search
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_

        mlflow.log_params(best_params)
        mlflow.log_metric(f"best_{scoring}_score", best_score)
        mlflow.sklearn.log_model(best_model, "model")

        print(f"\nBest Params: {best_params} | Best {scoring.upper()} Score: {best_score:.4f}")


# ==========================
# Main Execution
# ==========================
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare_data(
    "notebooks/data.csv"
)
    train_and_log_model(X_train, X_test, y_train, y_test)