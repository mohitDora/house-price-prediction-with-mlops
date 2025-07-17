# app/utils.py

import joblib
import pandas as pd
import numpy as np
import os, sys
from src.config import config
from src.logger import logger
from src.exception import MyException

def load_model_artifacts():
    """
    Loads the preprocessor and the trained model from joblib files.
    """
    logger.info("Attempting to load preprocessor and model artifacts...")
    try:
        preprocessor = joblib.load(config.PREPROCESSOR_PATH)
        logger.info(f"Preprocessor loaded from {config.PREPROCESSOR_PATH}")

        model = joblib.load(os.path.join("models", "final_model.joblib"))
        logger.info(f"Model loaded from models/final_model.joblib")

        logger.info("Model artifacts loaded successfully.")
        return preprocessor, model
    except FileNotFoundError as e:
        logger.error(f"Required model artifact not found: {e}", exc_info=True)
        raise MyException(f"Required model artifact not found: {e}", sys)
    except Exception as e:
        logger.error(f"Error loading model artifacts: {e}", exc_info=True)
        raise MyException(f"Error loading model artifacts: {e}", sys)

def convert_yes_no_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Converts 'yes'/'no' string columns to 1/0 integers in a DataFrame."""
    columns_to_convert = [
        'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea'
    ]
    for col in columns_to_convert:
        if col in df.columns:
            # 1. Map 'yes'/'no' to 1/0. This will introduce NaNs for other values.
            #    The series will likely become float dtype at this point if NaNs are introduced.
            mapped_series = df[col].map({'yes': 1, 'no': 0})

            # 2. Fill any remaining NaNs with 0.
            #    The series might still be an object dtype if the original column had mixed types,
            #    or float dtype if the map successfully yielded floats/NaNs.
            filled_series = mapped_series.fillna(0)

            # 3. Explicitly infer the best possible dtype *before* final type conversion.
            #    This is the step the FutureWarning is asking for when `fillna` might be operating on an object dtype.
            inferred_series = filled_series.infer_objects(copy=False)

            # 4. Convert to integer.
            df.loc[:, col] = inferred_series.astype(int)
            
    return df