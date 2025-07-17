# src/train_model.py

import json  # To save feature names for later use
import os
import sys

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.config import config
from src.exception import MyException
from src.logger import logger

# Set experiment name
mlflow.set_experiment("House Price Prediction")
mlflow.sklearn.autolog()


def load_processed_data_and_preprocessor(processed_data_path, preprocessor_path):
    """Loads processed data (X, y) and the preprocessor."""
    logger.info(f"Loading processed data from {processed_data_path}")
    try:
        processed_data = joblib.load(processed_data_path)
        X = processed_data["X"]
        y = processed_data["y"]

        logger.info(f"Loading preprocessor from {preprocessor_path}")
        preprocessor = joblib.load(preprocessor_path)

        logger.info(f"Loaded X shape: {X.shape}, y shape: {y.shape}")
        return X, y, preprocessor
    except Exception as e:
        logger.error(f"Failed to load processed data or preprocessor: {e}", exc_info=True)
        raise MyException(f"Failed to load processed data or preprocessor: {e}", sys)


def split_data(
    X,
    y,
    test_size=config.TEST_SPLIT_RATIO,
    val_size=config.VAL_SPLIT_RATIO,
    random_state=config.RANDOM_STATE,
):
    """Splits data into training, validation, and test sets using config values."""
    logger.info(f"Splitting data: test_size={test_size}, val_size={val_size}")
    try:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        relative_val_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=relative_val_size,
            random_state=random_state,
        )

        logger.info(f"Train set shape: {X_train.shape}")
        logger.info(f"Validation set shape: {X_val.shape}")
        logger.info(f"Test set shape: {X_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test
    except Exception as e:
        logger.error(f"Error during data splitting: {e}", exc_info=True)
        raise MyException(f"Error during data splitting: {e}", sys)


def train_model(
    X_train,
    y_train,
    n_estimators=config.RF_N_ESTIMATORS,
    max_features=config.RF_MAX_FEATURES,
    random_state=config.RANDOM_STATE,
):
    """Trains a RandomForestRegressor model using config values."""
    logger.info(f"Training RandomForestRegressor with n_estimators={n_estimators}, max_features={max_features}")
    try:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        logger.info("Model training complete.")
        return model
    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)
        raise MyException(f"Error during model training: {e}", sys)


def evaluate_model(model, X, y, dataset_name="validation"):
    """Evaluates the model and returns metrics."""
    logger.info(f"Evaluating model on {dataset_name} set.")
    try:
        predictions = model.predict(X)

        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)

        metrics = {
            f"{dataset_name}_mae": mae,
            f"{dataset_name}_rmse": rmse,
            f"{dataset_name}_r2": r2,
        }

        logger.info(f"--- {dataset_name} Metrics ---")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        return metrics
    except Exception as e:
        logger.error(f"Error during model evaluation on {dataset_name} set: {e}", exc_info=True)
        raise MyException(f"Error during model evaluation on {dataset_name} set: {e}", sys)


if __name__ == "__main__":
    logger.info("--- Starting Model Training Phase ---")
    try:
        # Ensure DVC pulls the latest data and preprocessor if not present or outdated
        logger.info("Ensuring latest data and preprocessor are available via DVC...")
        os.system(f"dvc pull {config.PROCESSED_DATA_PATH} {config.PREPROCESSOR_PATH}")
        logger.info("DVC pull complete.")

        # Load processed data and preprocessor
        X, y, preprocessor = load_processed_data_and_preprocessor(config.PROCESSED_DATA_PATH, config.PREPROCESSOR_PATH)

        # Define the expected input schema for the API.
        # This reflects the original raw input columns the user will provide.
        input_feature_schema = [
            {"name": "area", "type": "float"},
            {"name": "bedrooms", "type": "float"},
            {"name": "bathrooms", "type": "float"},
            {"name": "stories", "type": "float"},
            {"name": "mainroad", "type": "int", "options": [0, 1]},
            {"name": "guestroom", "type": "int", "options": [0, 1]},
            {"name": "basement", "type": "int", "options": [0, 1]},
            {"name": "hotwaterheating", "type": "int", "options": [0, 1]},
            {"name": "airconditioning", "type": "int", "options": [0, 1]},
            {"name": "parking", "type": "int", "options": [0, 1, 2, 3]},
            {"name": "prefarea", "type": "int", "options": [0, 1]},
            {
                "name": "furnishingstatus",
                "type": "str",
                "options": ["furnished", "semi-furnished", "unfurnished"],
            },
        ]

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

        with mlflow.start_run(run_name=f"RandomForest_N{config.RF_N_ESTIMATORS}_MF{config.RF_MAX_FEATURES}"):
            # Log parameters
            mlflow.log_param("model_type", "RandomForestRegressor")
            mlflow.log_param("n_estimators", config.RF_N_ESTIMATORS)
            mlflow.log_param("max_features", config.RF_MAX_FEATURES)
            mlflow.log_param("test_split_ratio", config.TEST_SPLIT_RATIO)
            mlflow.log_param("validation_split_ratio", config.VAL_SPLIT_RATIO)
            mlflow.log_param("random_state", config.RANDOM_STATE)

            # Train model
            model = train_model(X_train, y_train)

            # Evaluate model on validation set
            val_metrics = evaluate_model(model, X_val, y_val, dataset_name="validation")
            mlflow.log_metrics(val_metrics)

            # Evaluate model on test set (final unseen evaluation)
            test_metrics = evaluate_model(model, X_test, y_test, dataset_name="test")
            mlflow.log_metrics(test_metrics)

            # Log the scikit-learn model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=config.MODEL_ARTIFACT_PATH,
                registered_model_name=config.MODEL_NAME,
            )

            # Log the preprocessor as an artifact
            mlflow.log_artifact(config.PREPROCESSOR_PATH)

            # Log input feature schema as an artifact for later use by the serving API
            schema_file_name = "input_feature_schema.json"
            with open(schema_file_name, "w") as f:
                json.dump(input_feature_schema, f, indent=4)
            mlflow.log_artifact(schema_file_name)
            os.remove(schema_file_name)  # Clean up local file after logging

            joblib.dump(model, config.FINAL_MODEL_PATH)
            logger.info(f"Final trained model saved to {config.FINAL_MODEL_PATH}")
            # Log the local model artifact path to MLflow as well
            mlflow.log_artifact(config.FINAL_MODEL_PATH)

            logger.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
            logger.info("Model training and experiment logging complete.")

    except Exception as e:
        logger.error(f"An unexpected error occurred during model training: {e}", exc_info=True)
