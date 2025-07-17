# app/utils.py

# app/utils.py

import joblib
import os, sys
import subprocess # <--- NEW IMPORT
from src.config import config
from src.logger import logger
from src.exception import MyException

def load_model_artifacts():
    """
    Loads the preprocessor and the trained model from joblib files.
    Ensures DVC-tracked artifacts are pulled if not present.
    """
    logger.info("Attempting to load preprocessor and model artifacts...")
    try:
        preprocessor_path = config.PREPROCESSOR_PATH
        model_path = os.path.join("models", "final_model.joblib")

        # --- DVC PULL STEP ---
        logger.info(f"Checking for and attempting DVC pull for '{preprocessor_path}' and '{model_path}'...")
        try:
            # Use subprocess to run the dvc pull command
            # The 'check=True' will raise a CalledProcessError if the command fails
            # We explicitly pull each artifact to ensure they are present.
            subprocess.run(["dvc", "pull", preprocessor_path, model_path], check=True, capture_output=True, text=True)
            logger.info("DVC pull successful. Artifacts materialized.")
        except subprocess.CalledProcessError as e:
            logger.error(f"DVC pull failed with exit code {e.returncode}. STDOUT: {e.stdout}. STDERR: {e.stderr}", exc_info=True)
            raise MyException(f"DVC pull failed with exit code {e.returncode}")
        except FileNotFoundError:
            # This handles the case where the 'dvc' command itself is not found
            logger.error("DVC command not found. Is DVC installed and in PATH inside the container?", exc_info=True)
            raise MyException("DVC command not found. Is DVC installed and in PATH inside the container?")
        # --- END DVC PULL STEP ---

        # Now that DVC has pulled the files, they should exist locally in the container
        preprocessor = joblib.load(preprocessor_path)
        logger.info(f"Preprocessor loaded from {preprocessor_path}")

        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")

        logger.info("Model artifacts loaded successfully.")
        return preprocessor, model

    except FileNotFoundError as e:
        # This catch should ideally not be hit if DVC pull was successful
        logger.error(f"Required model artifact not found even after DVC pull attempt: {e}", exc_info=True)
        raise MyException(f"Required model artifact not found even after DVC pull attempt: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during model loading: {e}", exc_info=True)
        raise MyException(f"An unexpected error occurred during model loading: {e}")

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