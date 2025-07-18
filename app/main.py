# app/main.py

from fastapi import FastAPI, HTTPException, status
from pydantic import ValidationError
from typing import List
import pandas as pd
import numpy as np

from app.schema import HouseFeatures, PredictionResponse
from app.utils import load_model_artifacts, convert_yes_no_to_binary
from src.logger import logger
from src.exception import MyException # Make sure this is importable

app = FastAPI(
    title="House Price Prediction API",
    description="Predicts house prices based on various features.",
    version="0.1.0" # API version
)

# Global variables to store loaded model and preprocessor
preprocessor = None
model = None
model_version = "1.0.0" # A simple version for now, could be derived from MLflow run ID

@app.on_event("startup")
async def startup_event():
    """
    Load the model and preprocessor when the FastAPI application starts.
    """
    global preprocessor, model
    logger.info("Application startup: Loading model and preprocessor...")
    try:
        preprocessor, model = load_model_artifacts()
        # Optionally, get model version from MLflow if integrated robustly
        # For now, it's a static string
        logger.info("Model and preprocessor loaded successfully.")
    except Exception as e:
        logger.critical(f"An unexpected error occurred during model loading: {e}", exc_info=True)
        preprocessor = None
        model = None


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint to ensure the API is running.
    """
    return {"status": "ok", "message": "API is healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(features: HouseFeatures):
    """
    Predicts the price of a house based on the provided features.
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs for errors."
        )

    logger.info(f"Received prediction request: {features.model_dump()}")

    try:
        # Convert Pydantic model to a pandas DataFrame
        input_df = pd.DataFrame([features.model_dump()])

        # Ensure correct column order, which is critical for the preprocessor
        # The order should match the features list used during training
        expected_features_order = [
            'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom',
            'basement', 'hotwaterheating', 'airconditioning', 'parking',
            'prefarea', 'furnishingstatus'
        ]
        
        # Reindex to ensure correct order and handle any missing columns (e.g., if Optional was None)
        # Fill missing numeric values (bedrooms, area) with NaN to be handled by imputer in preprocessor
        for col in ['area', 'bedrooms']:
            if col not in input_df.columns or input_df[col].isnull().all():
                 input_df[col] = np.nan # Ensure numeric NaN for imputer

        input_df = input_df[expected_features_order]

        # 1. Convert 'yes'/'no' strings to 0/1 integers
        # This function handles the columns specified in app/utils.py
        input_df_converted = convert_yes_no_to_binary(input_df.copy())
        logger.debug(f"Input after yes/no conversion: {input_df_converted}")

        # 2. Apply the fitted preprocessor
        # It handles imputation, scaling, and one-hot encoding
        processed_input = preprocessor.transform(input_df_converted)
        logger.debug(f"Input after preprocessing (shape): {processed_input.shape}")

        # 3. Make prediction
        prediction = model.predict(processed_input)[0]
        logger.info(f"Prediction made: {prediction}")

        return PredictionResponse(predicted_price=round(float(prediction), 2), model_version=model_version)

    except ValidationError as e:
        logger.error(f"Input validation error: {e.errors()}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Validation Error", "errors": e.errors()}
        )
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during prediction: {e}"
        )