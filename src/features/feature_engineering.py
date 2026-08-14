# feature engineering
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import yaml
from src.logger import logging
import pickle


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


def apply_scaling(train_data: pd.DataFrame, test_data: pd.DataFrame, target_col: str = 'Churn') -> tuple:
    """Apply StandardScaler to the feature columns."""
    try:
        logging.info("Applying feature scaling...")
        scaler = StandardScaler()

        X_train = train_data.drop(columns=[target_col])
        y_train = train_data[target_col].values
        X_test = test_data.drop(columns=[target_col])
        y_test = test_data[target_col].values

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        train_df[target_col] = y_train

        test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        test_df[target_col] = y_test

        os.makedirs('models', exist_ok=True)
        pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
        logging.info('Feature scaling applied and data transformed')

        return train_df, test_df
    except Exception as e:
        logging.error('Error during feature scaling: %s', e)
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.info('Data saved to %s', file_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:
        params = load_params('params.yaml')
        target_col = params['feature_engineering'].get('target_col', 'Churn')

        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        train_df, test_df = apply_scaling(train_data, test_data, target_col)

        save_data(train_df, os.path.join("./data", "processed", "train_scaled.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_scaled.csv"))
    except Exception as e:
        logging.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()