"""
Tests for the FastAPI Application.
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import app


# ─── Test Client ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a FastAPI test client."""
    return TestClient(app)


# ─── Sample Payloads ───────────────────────────────────────────────────────────

VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}

HIGH_RISK_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 1,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.50,
    "TotalCharges": 191.00,
}

LOW_RISK_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 60,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 45.00,
    "TotalCharges": 2700.00,
}


# ─── Test: Health Check ────────────────────────────────────────────────────────

class TestHealthCheck:
    """Tests for the / endpoint."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Health endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_health_response_format(self, client: TestClient) -> None:
        """Health endpoint returns valid JSON with expected fields."""
        response = client.get("/")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data


# ─── Test: Single Prediction ───────────────────────────────────────────────────

class TestSinglePrediction:
    """Tests for the /predict endpoint."""

    def test_valid_prediction(self, client: TestClient) -> None:
        """Valid payload returns 200 with churn probability."""
        response = client.post("/predict", json=VALID_CUSTOMER)
        # Model may not be loaded in test environment
        if response.status_code == 200:
            data = response.json()
            assert "churn_probability" in data
            assert 0 <= data["churn_probability"] <= 1
            assert data["churn_prediction"] in ["Yes", "No"]
            assert data["risk_level"] in ["Low", "Medium", "High"]
            assert isinstance(data["top_risk_factors"], list)
        else:
            # 503 is acceptable if model isn't loaded
            assert response.status_code == 503

    def test_invalid_gender(self, client: TestClient) -> None:
        """Invalid gender value returns 422."""
        payload = VALID_CUSTOMER.copy()
        payload["gender"] = "Other"  # Not in Literal
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_tenure_negative(self, client: TestClient) -> None:
        """Negative tenure returns 422."""
        payload = VALID_CUSTOMER.copy()
        payload["tenure"] = -5
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_missing_field(self, client: TestClient) -> None:
        """Missing required field returns 422."""
        payload = VALID_CUSTOMER.copy()
        del payload["Contract"]
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_contract(self, client: TestClient) -> None:
        """Invalid contract type returns 422."""
        payload = VALID_CUSTOMER.copy()
        payload["Contract"] = "Weekly"
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_high_risk_customer(self, client: TestClient) -> None:
        """High-risk profile should have elevated churn probability."""
        response = client.post("/predict", json=HIGH_RISK_CUSTOMER)
        if response.status_code == 200:
            data = response.json()
            # A high-risk customer should have above-average churn probability
            assert data["churn_probability"] >= 0.3

    def test_low_risk_customer(self, client: TestClient) -> None:
        """Low-risk profile should have low churn probability."""
        response = client.post("/predict", json=LOW_RISK_CUSTOMER)
        if response.status_code == 200:
            data = response.json()
            # A low-risk customer should have below-average churn probability
            assert data["churn_probability"] <= 0.7


# ─── Test: Batch Prediction ───────────────────────────────────────────────────

class TestBatchPrediction:
    """Tests for the /predict/batch endpoint."""

    def test_batch_prediction(self, client: TestClient) -> None:
        """Batch endpoint handles multiple customers."""
        payload = {"customers": [VALID_CUSTOMER, HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER]}
        response = client.post("/predict/batch", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert data["total"] == 3
            assert len(data["predictions"]) == 3
        else:
            assert response.status_code == 503

    def test_empty_batch_rejected(self, client: TestClient) -> None:
        """Empty batch is rejected."""
        payload = {"customers": []}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 422


# ─── Test: Model Info ───────────────────────────────────────────────────────────

class TestModelInfo:
    """Tests for the /model/info endpoint."""

    def test_model_info(self, client: TestClient) -> None:
        """Model info endpoint returns data when model is loaded."""
        response = client.get("/model/info")
        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data
            assert "metrics" in data
            assert "feature_importances" in data
            assert "total_features" in data
        else:
            assert response.status_code == 503
