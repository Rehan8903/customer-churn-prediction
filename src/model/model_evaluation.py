import numpy as np
import pandas as pd
import pickle
import json
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import mlflow
import mlflow.sklearn
import dagshub
import os
from src.logger import logging


# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CHURN_PREDICTION_TOKEN")
if not dagshub_token:
    raise EnvironmentError("CHURN_PREDICTION_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "rehansarfraz8903"
repo_name = "Customer-Churn-prediction"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------

# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/rehansarfraz8903/Customer-Churn-prediction.mlflow')
# dagshub.init(repo_owner='rehansarfraz8903', repo_name='Customer-Churn-prediction', mlflow=True)
# -------------------------------------------------------------------------------------


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise


def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred).tolist()

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': cm
        }
        logging.info('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise


def save_model_info(run_id: str, model_uri: str, file_path: str) -> None:
    """Save the model run ID and URI to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        model_info = {'run_id': run_id, 'model_uri': model_uri}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise


def main():
    mlflow.set_experiment("churn-dvc-pipeline")
    with mlflow.start_run() as run:  # Start an MLflow run
        try:
            params = load_params('params.yaml')
            target_col = params['feature_engineering'].get('target_col', 'Churn')

            clf = load_model('./models/model.pkl')
            test_data = load_data('./data/processed/test_scaled.csv')

            X_test = test_data.drop(columns=[target_col]).values
            y_test = test_data[target_col].values

            metrics = evaluate_model(clf, X_test, y_test)

            save_metrics(metrics, 'reports/metrics.json')

            # Log metrics to MLflow (skip confusion_matrix — not a scalar)
            for metric_name, metric_value in metrics.items():
                if metric_name != 'confusion_matrix':
                    mlflow.log_metric(metric_name, metric_value)

            # Log model parameters to MLflow
            if hasattr(clf, 'get_params'):
                clf_params = clf.get_params()
                for param_name, param_value in clf_params.items():
                    mlflow.log_param(param_name, param_value)

            # Log model to MLflow — capture the returned info instead of assuming the path
            try:
                logged_model_info = mlflow.sklearn.log_model(clf, name="model")
            except TypeError:
    # Older MLflow versions (<2.16) don't support the `name` kwarg
                logged_model_info = mlflow.sklearn.log_model(clf, artifact_path="model")

            # Save model info using the actual URI MLflow gave us
            save_model_info(run.info.run_id, logged_model_info.model_uri, 'reports/experiment_info.json')

            # Log the metrics file to MLflow
            mlflow.log_artifact('reports/metrics.json')

        except Exception as e:
            logging.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")


if __name__ == '__main__':
    main()