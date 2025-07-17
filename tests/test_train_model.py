import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.config import config
from src.exception import MyException
from src.train_model import (
    evaluate_model,
    load_processed_data_and_preprocessor,
    split_data,
    train_model,
)


# Fixture for mock processed data and preprocessor
# Place this at the top of your test file
class MockPreprocessor:
    def transform(self, X):
        return X  # No transformation needed


@pytest.fixture(scope="module")
def mock_processed_artifacts(tmp_path_factory):
    """Creates mock processed data and preprocessor files for testing."""
    temp_dir = tmp_path_factory.mktemp("artifacts")

    # Create mock X (features) and y (target)
    X_mock = np.random.rand(100, 10)  # 100 samples, 10 features
    y_mock = np.random.rand(100) * 1_000_000  # Mock prices

    processed_data_path = temp_dir / "processed_test_data.parquet"
    preprocessor_path = temp_dir / "preprocessor_test.joblib"

    joblib.dump({"X": X_mock, "y": y_mock}, processed_data_path)
    joblib.dump(MockPreprocessor(), preprocessor_path)

    # Temporarily override config paths
    original_processed_data_path = config.PROCESSED_DATA_PATH
    original_preprocessor_path = config.PREPROCESSOR_PATH

    config.PROCESSED_DATA_PATH = str(processed_data_path)
    config.PREPROCESSOR_PATH = str(preprocessor_path)

    yield {
        "X": X_mock,
        "y": y_mock,
        "processed_data_path": processed_data_path,
        "preprocessor_path": preprocessor_path,
    }

    # Restore original config after tests
    config.PROCESSED_DATA_PATH = original_processed_data_path
    config.PREPROCESSOR_PATH = original_preprocessor_path


def test_load_processed_data_and_preprocessor(mock_processed_artifacts):
    """Test loading of processed data and preprocessor."""
    X, y, preprocessor = load_processed_data_and_preprocessor(
        mock_processed_artifacts["processed_data_path"],
        mock_processed_artifacts["preprocessor_path"],
    )

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert hasattr(preprocessor, "transform")  # Check if it behaves like a preprocessor
    assert np.array_equal(X, mock_processed_artifacts["X"])
    assert np.array_equal(y, mock_processed_artifacts["y"])


def test_load_processed_data_and_preprocessor_file_not_found(mock_processed_artifacts):
    """Test loading with non-existent files."""
    with pytest.raises(MyException, match="Failed to load processed data or preprocessor"):
        load_processed_data_and_preprocessor("non_existent_processed.parquet", "non_existent_preprocessor.joblib")


def test_split_data(mock_processed_artifacts):
    """Test data splitting proportions and types."""
    X_mock = mock_processed_artifacts["X"]
    y_mock = mock_processed_artifacts["y"]

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_mock, y_mock, test_size=0.2, val_size=0.1)

    total_samples = X_mock.shape[0]
    expected_test_samples = int(total_samples * 0.2)
    # val_size is relative to train_val split: 0.1 / (1 - 0.2) = 0.125
    expected_val_samples = int((total_samples - expected_test_samples) * (0.1 / (1 - 0.2)))
    expected_train_samples = total_samples - expected_test_samples - expected_val_samples

    assert X_train.shape[0] == expected_train_samples
    assert X_val.shape[0] == expected_val_samples
    assert X_test.shape[0] == expected_test_samples

    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)


def test_train_model(mock_processed_artifacts):
    """Test model training and return type."""
    X_mock = mock_processed_artifacts["X"]
    y_mock = mock_processed_artifacts["y"]

    # Use a small subset for quick testing
    X_train_small = X_mock[:10]
    y_train_small = y_mock[:10]

    model = train_model(X_train_small, y_train_small, n_estimators=5, max_features=2)  # Use small params for speed
    assert isinstance(model, RandomForestRegressor)
    assert hasattr(model, "predict")
    assert model.n_estimators == 5


def test_evaluate_model(mock_processed_artifacts):
    """Test model evaluation and metrics returned."""
    X_mock = mock_processed_artifacts["X"]
    y_mock = mock_processed_artifacts["y"]

    # Create a dummy model for evaluation
    class MockModel:
        def predict(self, X):
            return np.full(X.shape[0], y_mock.mean())  # Predict mean for simplicity

    model = MockModel()
    metrics = evaluate_model(model, X_mock, y_mock, dataset_name="test_dataset")

    assert isinstance(metrics, dict)
    assert "test_dataset_mae" in metrics
    assert "test_dataset_rmse" in metrics
    assert "test_dataset_r2" in metrics
    assert metrics["test_dataset_mae"] >= 0
    assert metrics["test_dataset_rmse"] >= 0
    # R2 can be negative for poor models, but for a constant prediction like mean, it's 0
    assert abs(metrics["test_dataset_r2"]) < 0.001  # Should be close to 0 for mean prediction
