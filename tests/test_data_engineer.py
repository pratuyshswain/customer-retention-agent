"""
Tests for the Data Engineering Module.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_dataset import generate_telco_dataset
from src.data_engineer import clean_data, build_preprocessor, load_raw_data


# ─── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    """Generate a fresh synthetic dataset for testing."""
    return generate_telco_dataset(n_rows=500, seed=99)


@pytest.fixture(scope="module")
def cleaned_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Return cleaned version of the raw dataset."""
    return clean_data(raw_df)


@pytest.fixture(scope="module")
def processed_result(cleaned_df: pd.DataFrame) -> tuple:
    """Return processed DataFrame + preprocessor + feature columns."""
    return build_preprocessor(cleaned_df)


# ─── Test: Data Generation ──────────────────────────────────────────────────────

class TestDataGeneration:
    """Tests for the synthetic dataset generator."""

    def test_shape(self, raw_df: pd.DataFrame) -> None:
        """Dataset has the correct number of columns."""
        assert raw_df.shape[1] == 21, f"Expected 21 columns, got {raw_df.shape[1]}"

    def test_row_count(self, raw_df: pd.DataFrame) -> None:
        """Dataset has the requested number of rows."""
        assert raw_df.shape[0] == 500

    def test_column_names(self, raw_df: pd.DataFrame) -> None:
        """All expected columns are present."""
        expected_cols = {
            "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
            "tenure", "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
            "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
        }
        assert set(raw_df.columns) == expected_cols

    def test_churn_values(self, raw_df: pd.DataFrame) -> None:
        """Churn column contains only 'Yes' and 'No'."""
        assert set(raw_df["Churn"].unique()) == {"Yes", "No"}

    def test_churn_rate_approximate(self, raw_df: pd.DataFrame) -> None:
        """Churn rate is approximately 26.5% (±10%)."""
        churn_rate = (raw_df["Churn"] == "Yes").mean()
        assert 0.16 < churn_rate < 0.37, f"Churn rate {churn_rate:.2%} is out of range"

    def test_unique_customer_ids(self, raw_df: pd.DataFrame) -> None:
        """All customer IDs are unique."""
        assert raw_df["customerID"].nunique() == len(raw_df)


# ─── Test: Data Cleaning ───────────────────────────────────────────────────────

class TestDataCleaning:
    """Tests for the data cleaning pipeline."""

    def test_no_nulls_after_cleaning(self, cleaned_df: pd.DataFrame) -> None:
        """No null values remain after cleaning."""
        assert cleaned_df.isnull().sum().sum() == 0

    def test_customer_id_dropped(self, cleaned_df: pd.DataFrame) -> None:
        """customerID column is removed."""
        assert "customerID" not in cleaned_df.columns

    def test_churn_is_numeric(self, cleaned_df: pd.DataFrame) -> None:
        """Churn is encoded as 0/1 integers."""
        assert cleaned_df["Churn"].dtype in [np.int64, np.int32, int]
        assert set(cleaned_df["Churn"].unique()) == {0, 1}

    def test_total_charges_numeric(self, cleaned_df: pd.DataFrame) -> None:
        """TotalCharges is now a float column."""
        assert cleaned_df["TotalCharges"].dtype in [np.float64, np.float32, float]


# ─── Test: Feature Engineering ──────────────────────────────────────────────────

class TestFeatureEngineering:
    """Tests for the preprocessing pipeline."""

    def test_all_numeric(self, processed_result: tuple) -> None:
        """All features are numeric after encoding."""
        df, _, _ = processed_result
        non_numeric = df.select_dtypes(include=["object"]).columns.tolist()
        assert len(non_numeric) == 0, f"Non-numeric columns remain: {non_numeric}"

    def test_no_nulls(self, processed_result: tuple) -> None:
        """No null values in processed data."""
        df, _, _ = processed_result
        assert df.isnull().sum().sum() == 0

    def test_scaled_features_mean(self, processed_result: tuple) -> None:
        """Scaled numerical features have mean ≈ 0."""
        df, _, _ = processed_result
        for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
            mean = df[col].mean()
            assert abs(mean) < 0.2, f"{col} mean is {mean:.4f}, expected ≈ 0"

    def test_scaled_features_std(self, processed_result: tuple) -> None:
        """Scaled numerical features have std ≈ 1."""
        df, _, _ = processed_result
        for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
            std = df[col].std()
            assert 0.5 < std < 1.5, f"{col} std is {std:.4f}, expected ≈ 1"

    def test_preprocessor_has_required_keys(self, processed_result: tuple) -> None:
        """Preprocessor bundle contains all required components."""
        _, preprocessor, _ = processed_result
        required_keys = {
            "binary_mapping", "service_mapping", "service_binary_cols",
            "multiclass_cols", "numerical_cols", "ohe", "scaler",
        }
        assert required_keys.issubset(set(preprocessor.keys()))

    def test_feature_columns_returned(self, processed_result: tuple) -> None:
        """Feature column names are returned."""
        _, _, feature_cols = processed_result
        assert len(feature_cols) > 0
        assert "Churn" not in feature_cols
