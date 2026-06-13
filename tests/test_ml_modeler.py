"""
Tests for the ML Modeling Module.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_dataset import generate_telco_dataset
from src.data_engineer import clean_data, build_preprocessor
from src.ml_modeler import (
    evaluate_model,
    get_feature_importances,
    split_data,
    train_logistic_regression,
    train_random_forest,
)


# ─── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def processed_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Generate and preprocess data for modeling tests."""
    raw_df = generate_telco_dataset(n_rows=1000, seed=123)
    cleaned = clean_data(raw_df)
    processed, _, feature_cols = build_preprocessor(cleaned)
    X = processed[feature_cols]
    y = processed["Churn"]
    return X, y, feature_cols


@pytest.fixture(scope="module")
def split_result(
    processed_data: tuple[pd.DataFrame, pd.Series, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train/test split."""
    X, y, _ = processed_data
    return split_data(X, y)


@pytest.fixture(scope="module")
def trained_lr(
    split_result: tuple,
) -> LogisticRegression:
    """Trained Logistic Regression model."""
    X_train, _, y_train, _ = split_result
    return train_logistic_regression(X_train, y_train)


@pytest.fixture(scope="module")
def trained_rf(
    split_result: tuple,
) -> RandomForestClassifier:
    """Trained Random Forest model (small grid for speed)."""
    X_train, _, y_train, _ = split_result
    return train_random_forest(X_train, y_train)


# ─── Test: Train/Test Split ────────────────────────────────────────────────────

class TestTrainTestSplit:
    """Tests for the stratified train/test split."""

    def test_split_sizes(self, split_result: tuple) -> None:
        """Train and test sets have the correct relative sizes."""
        X_train, X_test, _, _ = split_result
        total = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total
        assert 0.15 < test_ratio < 0.25, f"Test ratio {test_ratio:.2%} out of range"

    def test_stratification(self, split_result: tuple) -> None:
        """Churn ratio is preserved (±2%) between train and test."""
        _, _, y_train, y_test = split_result
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        diff = abs(train_rate - test_rate)
        assert diff < 0.02, (
            f"Class ratio not preserved: train={train_rate:.3f}, test={test_rate:.3f}"
        )

    def test_no_data_leakage(self, split_result: tuple) -> None:
        """Train and test indices don't overlap."""
        X_train, X_test, _, _ = split_result
        train_idx = set(X_train.index)
        test_idx = set(X_test.index)
        assert len(train_idx & test_idx) == 0, "Data leakage detected!"


# ─── Test: Logistic Regression ──────────────────────────────────────────────────

class TestLogisticRegression:
    """Tests for the LR baseline model."""

    def test_model_type(self, trained_lr: LogisticRegression) -> None:
        """Model is a LogisticRegression instance."""
        assert isinstance(trained_lr, LogisticRegression)

    def test_probabilities_valid(
        self, trained_lr: LogisticRegression, split_result: tuple
    ) -> None:
        """Predicted probabilities are in [0, 1]."""
        _, X_test, _, _ = split_result
        probs = trained_lr.predict_proba(X_test)[:, 1]
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_roc_auc_minimum(
        self, trained_lr: LogisticRegression, split_result: tuple
    ) -> None:
        """ROC-AUC is at least 0.65 (lenient for small dataset)."""
        _, X_test, _, y_test = split_result
        probs = trained_lr.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        assert auc >= 0.65, f"LR ROC-AUC {auc:.4f} < 0.65"


# ─── Test: Random Forest ───────────────────────────────────────────────────────

class TestRandomForest:
    """Tests for the Random Forest model."""

    def test_model_type(self, trained_rf: RandomForestClassifier) -> None:
        """Model is a RandomForestClassifier instance."""
        assert isinstance(trained_rf, RandomForestClassifier)

    def test_probabilities_valid(
        self, trained_rf: RandomForestClassifier, split_result: tuple
    ) -> None:
        """Predicted probabilities are in [0, 1]."""
        _, X_test, _, _ = split_result
        probs = trained_rf.predict_proba(X_test)[:, 1]
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_roc_auc_minimum(
        self, trained_rf: RandomForestClassifier, split_result: tuple
    ) -> None:
        """ROC-AUC is at least 0.75."""
        _, X_test, _, y_test = split_result
        probs = trained_rf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        assert auc >= 0.75, f"RF ROC-AUC {auc:.4f} < 0.75"


# ─── Test: Feature Importances ──────────────────────────────────────────────────

class TestFeatureImportances:
    """Tests for feature importance extraction."""

    def test_rf_importances(
        self,
        trained_rf: RandomForestClassifier,
        processed_data: tuple[pd.DataFrame, pd.Series, list[str]],
    ) -> None:
        """Random Forest produces feature importances."""
        _, _, feature_cols = processed_data
        importances = get_feature_importances(trained_rf, feature_cols, top_n=5)
        assert len(importances) == 5
        for name, val in importances:
            assert isinstance(name, str)
            assert isinstance(val, float)
            assert val >= 0

    def test_lr_importances(
        self,
        trained_lr: LogisticRegression,
        processed_data: tuple[pd.DataFrame, pd.Series, list[str]],
    ) -> None:
        """Logistic Regression produces coefficient-based importances."""
        _, _, feature_cols = processed_data
        importances = get_feature_importances(trained_lr, feature_cols, top_n=5)
        assert len(importances) == 5


# ─── Test: Evaluation Metrics ───────────────────────────────────────────────────

class TestEvaluationMetrics:
    """Tests for the evaluation function."""

    def test_metrics_keys(
        self, trained_rf: RandomForestClassifier, split_result: tuple
    ) -> None:
        """Evaluation returns all expected metric keys."""
        _, X_test, _, y_test = split_result
        metrics = evaluate_model(trained_rf, X_test, y_test, "RF")
        expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc"}
        assert set(metrics.keys()) == expected_keys

    def test_metrics_range(
        self, trained_rf: RandomForestClassifier, split_result: tuple
    ) -> None:
        """All metrics are in [0, 1]."""
        _, X_test, _, y_test = split_result
        metrics = evaluate_model(trained_rf, X_test, y_test, "RF")
        for name, val in metrics.items():
            assert 0 <= val <= 1, f"{name} = {val} is out of [0, 1]"
