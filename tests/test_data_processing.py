# tests/test_data_processing.py

import pytest
import pandas as pd
import numpy as np
import os
import joblib
from src.data_processing import DataProcessor
from src.exception import MyException
from src.config import config

# Fixture to provide a temporary raw data CSV for testing
@pytest.fixture(scope="module")
def temp_raw_data_csv(tmp_path_factory):
    """
    Creates a temporary raw_data.csv with mixed 'yes'/'no' and 0/1,
    and also some missing values for testing.
    """
    temp_dir = tmp_path_factory.mktemp("data")
    temp_csv_path = temp_dir / "raw_data.csv"

    # Sample data that matches your described format, including 'yes'/'no'
    # and some missing values and potential edge cases for testing
    data = {
        'price': [13300000, 12250000, 12250000, 5000000, 6500000, 7800000, 4000000, 9000000, 5500000, 7000000],
        'area': [7420, 8960, 9960, 3000, 4500, 5500, 3200, 6800, 4100, 5000], # NaN for area
        'bedrooms': [4, 4, 3, 2, 3, 2, 2, 3, 2, 3], # NaN for bedrooms
        'bathrooms': [2, 4, 2, 1, 2, 2, 1, 3, 1, 2],
        'stories': [3, 4, 2, 1, 2, 2, 1, 3, 1, 2],
        'mainroad': ['yes', 'yes', 'yes', 'no', 'yes', 'yes', 'no', 'yes', 'no', 'yes'],
        'guestroom': ['no', 'no', 'no', 'no', 'yes', 'no', 'no', 'yes', 'no', 'no'],
        'basement': ['no', 'no', 'yes', 'no', 'no', 'yes', 'no', 'yes', 'no', 'no'],
        'hotwaterheating': ['no', 'no', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'no'],
        'airconditioning': ['yes', 'yes', 'no', 'no', 'yes', 'no', 'no', 'yes', 'no', 'yes'],
        'parking': [2, 3, 2, 0, 1, 1, 0, 2, 0, 1],
        'prefarea': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes'],
        'furnishingstatus': ['furnished', 'furnished', 'semi-furnished', 'unfurnished', 'semi-furnished', 'furnished', 'unfurnished', 'furnished', 'semi-furnished', 'furnished']
    }
    df = pd.DataFrame(data)
    df.to_csv(temp_csv_path, index=False)
    
    # Temporarily override config's RAW_DATA_PATH
    original_raw_data_path = config.RAW_DATA_PATH
    config.RAW_DATA_PATH = str(temp_csv_path)

    yield temp_csv_path

    # Restore original config after tests are done
    config.RAW_DATA_PATH = original_raw_data_path
    # Clean up temp directory
    # tmp_path_factory manages cleanup for mktemp

@pytest.fixture(scope="module")
def data_processor_instance(temp_raw_data_csv):
    """Provides a DataProcessor instance configured for testing."""
    # Ensure paths are set correctly for the fixture scope
    processor = DataProcessor()
    processor.raw_data_path = str(temp_raw_data_csv) # Ensure processor uses temp path
    processor.processed_data_path = str(temp_raw_data_csv.parent / "processed_test_data.parquet")
    processor.preprocessor_path = str(temp_raw_data_csv.parent / "preprocessor_test.joblib")
    return processor

def test_load_data(data_processor_instance):
    print("processor",data_processor_instance)
    print("processor",data_processor_instance.processed_data_path)
    print("processor",data_processor_instance.preprocessor_path)
    print("processor",data_processor_instance.raw_data_path)
    """Test if data loads correctly."""
    df = data_processor_instance.load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'price' in df.columns
    assert df.shape[0] == 10 # Based on sample data length

def test_load_data_file_not_found(data_processor_instance):
    """Test loading data from a non-existent path."""
    original_path = data_processor_instance.raw_data_path
    data_processor_instance.raw_data_path = "non_existent_file.csv"
    with pytest.raises(MyException, match="Failed to load raw data"):
        data_processor_instance.load_data()
    data_processor_instance.raw_data_path = original_path # Restore for other tests

def test_convert_yes_no_to_binary(data_processor_instance):
    """Test conversion of 'yes'/'no' strings to 0/1 integers."""
    df = data_processor_instance.load_data()
    columns_to_convert = [
        'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea'
    ]
    converted_df = data_processor_instance._convert_yes_no_to_binary(df.copy(), columns_to_convert)

    for col in columns_to_convert:
        assert pd.api.types.is_integer_dtype(converted_df[col]), f"{col} is not integer type"
        assert converted_df[col].isin([0, 1]).all(), f"{col} contains values other than 0 or 1"
    
    # Specific check for a known conversion
    assert converted_df.loc[0, 'mainroad'] == 1 # 'yes'
    assert converted_df.loc[3, 'mainroad'] == 0 # 'no'
    assert converted_df.loc[2, 'basement'] == 1 # 'yes'

def test_validate_raw_data_schema_success(data_processor_instance):
    """Test schema validation on correctly converted data."""
    df = data_processor_instance.load_data()
    columns_to_convert = [
        'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea'
    ]
    converted_df = data_processor_instance._convert_yes_no_to_binary(df.copy(), columns_to_convert)
    
    try:
        validated_df = data_processor_instance.validate_raw_data(converted_df)
        assert isinstance(validated_df, pd.DataFrame)
        assert validated_df.shape == converted_df.shape # Shape shouldn't change
    except Exception as e:
        pytest.fail(f"Schema validation failed unexpectedly: {e}")


def test_validate_raw_data_schema_failure(data_processor_instance):
    """Test schema validation with invalid data (e.g., wrong type for price)."""
    df_invalid = data_processor_instance.load_data()
    
    # Introduce an invalid type (e.g., string in price)
    df_invalid.loc[0, 'price'] = "not_a_number"
    
    # Also ensure booleans are converted before validation for this test to be accurate on other cols
    columns_to_convert = [
        'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea'
    ]
    df_invalid = data_processor_instance._convert_yes_no_to_binary(df_invalid, columns_to_convert)

    with pytest.raises(MyException,match="Raw data schema validation failed"):
        data_processor_instance.validate_raw_data(df_invalid)

def test_preprocess_data(data_processor_instance):
    """Test end-to-end preprocessing."""
    df = data_processor_instance.load_data()
    X_processed, y, preprocessor, features, target = data_processor_instance.preprocess_data(df.copy()) # Pass copy

    assert isinstance(X_processed, np.ndarray)
    assert isinstance(y, pd.Series)
    assert y.name == 'price'
    assert 'sklearn.compose._column_transformer.ColumnTransformer' in str(type(preprocessor))
    assert len(features) == 12 # Ensure all 12 input features are used
    
    # Check if NaNs were handled (imputed) in X_processed
    assert not pd.isnull(X_processed).any()

    # Check output shape. For 10 samples, 12 features.
    # Numerical features (5) + binary (6) + one-hot for furnishingstatus (3) = 14 columns
    assert X_processed.shape == (10, 14) # 10 samples - 1 duplicate = 9
    
    # Verify that the duplicate handling worked
    assert df.drop_duplicates().shape[0] == 10 # Assuming one duplicate from test data

def test_save_and_load_processed_artifacts(data_processor_instance):
    """Test saving and loading processed data and preprocessor."""
    df = data_processor_instance.load_data()
    X_processed, y, preprocessor, _, _ = data_processor_instance.preprocess_data(df.copy())

    data_processor_instance.save_processed_artifacts(X_processed, y, preprocessor)

    # Check if files exist
    assert os.path.exists(data_processor_instance.processed_data_path)
    assert os.path.exists(data_processor_instance.preprocessor_path)

    # Load and verify content
    loaded_processed_data = joblib.load(data_processor_instance.processed_data_path)
    loaded_preprocessor = joblib.load(data_processor_instance.preprocessor_path)

    assert np.array_equal(loaded_processed_data['X'], X_processed)
    assert np.array_equal(loaded_processed_data['y'], y.values)
    assert 'sklearn.compose._column_transformer.ColumnTransformer' in str(type(loaded_preprocessor))

    # Clean up test artifacts
    os.remove(data_processor_instance.processed_data_path)
    os.remove(data_processor_instance.preprocessor_path)