import os

from dotenv import load_dotenv

from src.logger import logger

# Load environment variables from .env file
load_dotenv()


class Config:
    """Centralized configuration for the project."""

    # MongoDB Configuration
    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
    MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME")

    # Data Paths
    RAW_DATA_PATH = "data/raw_data.csv"
    PROCESSED_DATA_PATH = "data/processed_data.parquet"
    PREPROCESSOR_PATH = "models/preprocessor.joblib"
    FINAL_MODEL_PATH = "models/final_model.joblib"

    # Model Configuration
    MODEL_NAME = "HousePricePredictor"
    MODEL_ARTIFACT_PATH = "house_price_model"

    # Training Parameters
    TEST_SPLIT_RATIO = 0.2
    VAL_SPLIT_RATIO = 0.1
    RANDOM_STATE = 42

    # RandomForestRegressor specific parameters
    RF_N_ESTIMATORS = int(os.getenv("RF_N_ESTIMATORS", 100))
    RF_MAX_FEATURES = int(os.getenv("RF_MAX_FEATURES", 5))

    # FastAPI Server Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))

    def __init__(self):
        # Basic validation for essential configs
        if not self.MONGO_URI:
            logger.error("MONGO_URI is not set in environment variables or .env file.")
            raise ValueError("MONGO_URI must be set.")
        logger.info("Configuration loaded successfully.")


# Initialize config
config = Config()

if __name__ == "__main__":
    # Example usage and verification
    logger.info(f"MongoDB URI: {config.MONGO_URI}")
    logger.info(f"Raw Data Path: {config.RAW_DATA_PATH}")
    logger.info(f"Random Forest N_ESTIMATORS: {config.RF_N_ESTIMATORS}")
