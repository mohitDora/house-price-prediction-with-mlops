import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib  # For saving preprocessor
import pandera.pandas as pa
from sklearn.pipeline import Pipeline
from pandera.errors import SchemaErrors
import sys

from src.logger import logger
from src.exception import MyException
from src.config import config


class DataProcessor:
    def __init__(self):
        self.raw_data_path = config.RAW_DATA_PATH
        self.processed_data_path = config.PROCESSED_DATA_PATH
        self.preprocessor_path = config.PREPROCESSOR_PATH

    def load_data(self):
        """Loads data from the raw CSV file."""
        logger.info(f"Loading raw data from {self.raw_data_path}")
        try:
            df = pd.read_csv(self.raw_data_path)
            logger.info(f"Loaded data. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(
                f"Failed to load raw data from {self.raw_data_path}: {e}", exc_info=True
            )
            raise MyException(f"Failed to load raw data from {self.raw_data_path}", sys)

    def _convert_yes_no_to_binary(self, df, columns):
        """Converts 'yes'/'no' to 1/0 and ensures final dtype is int."""
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().map({"yes": 1, "no": 0}).fillna(0).astype(int)
                logger.debug(f"Converted 'yes'/'no' to 1/0 in column: {col}")
        return df


    def validate_raw_data(self, df):
        """Validates the raw data schema using Pandera based on the new headers."""
        logger.info("Validating raw data schema.")
        schema = pa.DataFrameSchema(
            {
                "price": pa.Column(pa.Int, checks=pa.Check.gt(0)),
                "area": pa.Column(pa.Int, nullable=True, checks=pa.Check.gt(0)),
                "bedrooms": pa.Column(pa.Int, nullable=True, checks=pa.Check.ge(1)),
                "bathrooms": pa.Column(pa.Int, checks=pa.Check.ge(1)),
                "stories": pa.Column(pa.Int, checks=pa.Check.ge(1)),
                "mainroad": pa.Column(pa.Int, checks=pa.Check.isin([0, 1])),
                "guestroom": pa.Column(pa.Int, checks=pa.Check.isin([0, 1])),
                "basement": pa.Column(pa.Int, checks=pa.Check.isin([0, 1])),
                "hotwaterheating": pa.Column(pa.Int, checks=pa.Check.isin([0, 1])),
                "airconditioning": pa.Column(pa.Int, checks=pa.Check.isin([0, 1])),
                "parking": pa.Column(pa.Int, checks=pa.Check.ge(0)),
                "prefarea": pa.Column(pa.Int, checks=pa.Check.isin([0, 1])),
                "furnishingstatus": pa.Column(
                    pa.String,
                    checks=pa.Check.isin(
                        ["furnished", "semi-furnished", "unfurnished"]
                    ),
                ),
            }
        )
        try:
            validated_df = schema.validate(df, lazy=True)
            logger.info("Raw data schema validation successful.")
            return validated_df
        except SchemaErrors as err:
            logger.error(
                f"Raw data schema validation failed: {err.failure_cases}", exc_info=True
            )
            raise MyException(
                f"Raw data schema validation failed: {err.failure_cases}", sys
            )
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during raw data validation: {e}",
                exc_info=True,
            )
            raise MyException(
                f"An unexpected error occurred during raw data validation: {e}", sys
            )

    def preprocess_data(self, df):
        """Cleans data, engineers features, and creates a preprocessor for the new dataset."""
        logger.info("Starting comprehensive data preprocessing...")
        try:
            # 1. Handle 'yes'/'no' to 1/0 conversion (FIRST STEP)
            # columns_to_convert = [
            #     'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea'
            # ]
            # df = self._convert_yes_no_to_binary(df, columns_to_convert)
            
            # 2. Drop potential duplicate rows
            df.drop_duplicates(inplace=True)
            logger.debug(f"Dropped duplicates. New shape: {df.shape}")

            # Define features and target
            features = [
                'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom',
                'basement', 'hotwaterheating', 'airconditioning', 'parking',
                'prefarea', 'furnishingstatus'
            ]
            target = 'price'

            X = df[features]
            y = df[target]

            numerical_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
            # These are already 0/1 integers after conversion, so they behave numerically
            binary_categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
            
            multi_categorical_features = ['furnishingstatus']

            # Create transformers
            numerical_transformer = ColumnTransformer(
                transformers=[
                    ('impute_and_scale', 
                     Pipeline([
                         ('imputer', SimpleImputer(strategy='median')),
                         ('scaler', StandardScaler())
                     ]), 
                     numerical_features)
                ],
                remainder='passthrough'
            )

            multi_categorical_transformer = OneHotEncoder(handle_unknown='ignore')

            # Create a combined preprocessor pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipe', numerical_transformer, numerical_features),
                    ('binary_cat', 'passthrough', binary_categorical_features),
                    ('multi_cat_enc', multi_categorical_transformer, multi_categorical_features)
                ]
            )

            preprocessor.fit(X)
            X_processed = preprocessor.transform(X)
            
            logger.info("Data preprocessing complete. Preprocessor fitted and data transformed.")
            return X_processed, y, preprocessor, features, target

        except Exception as e:
            logger.error(f"Error during data preprocessing: {e}", exc_info=True)
            raise MyException(f"Error during data preprocessing: {e}", sys)

    def save_processed_artifacts(self, X_processed, y, preprocessor):
        """Saves processed data and the preprocessor."""
        try:
            processed_data = {
                'X': X_processed,
                'y': y.values
            }
            joblib.dump(processed_data, self.processed_data_path)
            joblib.dump(preprocessor, self.preprocessor_path)
            logger.info(f"Processed data saved to {self.processed_data_path}")
            logger.info(f"Preprocessor saved to {self.preprocessor_path}")
        except Exception as e:
            logger.error(f"Failed to save processed artifacts: {e}", exc_info=True)
            raise MyException(f"Failed to save processed artifacts: {e}", sys)
        
if __name__ == "__main__":
    logger.info("--- Starting Data Processing Phase (via DataProcessor) ---")
    data_processor = DataProcessor()
    try:        
        # Load raw data
        raw_df_from_csv = data_processor.load_data()

        # Perform the yes/no conversion within the main script for validation clarity
        intermediate_df_for_validation = data_processor._convert_yes_no_to_binary(raw_df_from_csv.copy(), [
            'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea'
        ])

        # Validate the schema of the converted raw data
        validated_raw_df = data_processor.validate_raw_data(intermediate_df_for_validation)

        # Preprocess the data
        X_processed, y, preprocessor, _, _ = data_processor.preprocess_data(validated_raw_df.copy()) # Pass a copy

        # Save processed data and preprocessor
        data_processor.save_processed_artifacts(X_processed, y, preprocessor)

        logger.info("--- Data Processing Phase Complete ---")

    except Exception as e:
        logger.error(f"An unexpected error occurred during data processing: {e}", exc_info=True)