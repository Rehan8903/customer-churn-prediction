# register model

import json
import mlflow
from src.logger import logging
import os
import dagshub

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

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


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise


def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry and alias it as 'staging'."""
    try:
        model_uri = model_info['model_uri']  # use the exact URI MLflow gave us, don't reconstruct it

        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)

        client = mlflow.tracking.MlflowClient()

        # Set an alias instead of the deprecated stage transition
        client.set_registered_model_alias(
            name=model_name,
            alias="staging",
            version=model_version.version
        )

        logging.debug(
            f'Model {model_name} version {model_version.version} registered and aliased as "staging".'
        )
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise


def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = "churn_prediction_model"
        register_model(model_name, model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()