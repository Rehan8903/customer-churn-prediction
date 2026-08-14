# data preprocessing

import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from src.logger import logging


def preprocess_dataframe(df: pd.DataFrame, encoders: dict = None, fit: bool = True) -> tuple:
    """
    Preprocess a churn DataFrame by label-encoding categorical columns.

    Args:
        df (pd.DataFrame): The DataFrame to preprocess.
        encoders (dict): Dict of {column: LabelEncoder} to reuse (for test data).
                          If None and fit=True, new encoders are created.
        fit (bool): Whether to fit new encoders (True for train) or reuse
                     passed-in encoders (False for test).

    Returns:
        tuple: (processed DataFrame, dict of fitted encoders)
    """
    try:
        df = df.copy()

        # All non-numeric columns are treated as categorical
        categorical_cols = df.select_dtypes(include='object').columns.tolist()

        if encoders is None:
            encoders = {}

        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
            else:
                le = encoders[col]
                # Handle unseen categories in test data gracefully instead of crashing
                df[col] = df[col].astype(str).map(
                    lambda val: le.transform([val])[0] if val in le.classes_ else -1
                )

        logging.info("Data pre-processing completed")
        return df, encoders

    except Exception as e:
        logging.error('Error during preprocessing: %s', e)
        raise


def main():
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logging.info('data loaded properly')

        # Fit encoders on train, reuse the same encoders on test
        train_processed_data, encoders = preprocess_dataframe(train_data, fit=True)
        test_processed_data, _ = preprocess_dataframe(test_data, encoders=encoders, fit=False)

        # Store the data inside data/interim
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

        # Save fitted encoders so inference/serving code can reuse them
        models_path = os.path.join("./models")
        os.makedirs(models_path, exist_ok=True)
        with open(os.path.join(models_path, "label_encoders.pkl"), "wb") as f:
            pickle.dump(encoders, f)

        logging.info('Processed data saved to %s', data_path)
        logging.info('Label encoders saved to %s', models_path)

    except Exception as e:
        logging.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()