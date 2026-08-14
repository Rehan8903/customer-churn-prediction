import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import yaml
from src.logger import logging


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


def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> LogisticRegression:
    """Train the Logistic Regression model."""
    try:
        clf = LogisticRegression(
            C=params.get('C', 1),
            solver=params.get('solver', 'liblinear'),
            penalty=params.get('penalty', 'l1'),
            class_weight=params.get('class_weight', None),
            random_state=42
        )
        clf.fit(X_train, y_train)
        logging.info('Model training completed')
        return clf
    except Exception as e:
        logging.error('Error during model training: %s', e)
        raise


def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise


def main():
    try:
        params = load_params('params.yaml')
        model_params = params['model_training']
        target_col = params['feature_engineering'].get('target_col', 'Churn')

        train_data = load_data('./data/processed/train_scaled.csv')
        X_train = train_data.drop(columns=[target_col]).values
        y_train = train_data[target_col].values

        clf = train_model(X_train, y_train, model_params)

        save_model(clf, 'models/model.pkl')
    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()