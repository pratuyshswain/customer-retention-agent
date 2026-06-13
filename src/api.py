"""
FastAPI Application — Customer Churn Prediction API.

Endpoints:
  GET  /            → Health check + model info
  POST /predict     → Single customer churn probability
  POST /predict/batch → Batch prediction
  GET  /model/info  → Model metrics & feature importances
"""

import os
from contextlib import asynccontextmanager
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ─── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEST_MODEL_PATH: str = os.path.join(BASE_DIR, "models", "best_model.joblib")
PREPROCESSOR_PATH: str = os.path.join(BASE_DIR, "models", "preprocessor.joblib")
MODEL_METADATA_PATH: str = os.path.join(BASE_DIR, "models", "model_metadata.joblib")
FEATURE_COLUMNS_PATH: str = os.path.join(BASE_DIR, "models", "feature_columns.joblib")

# ─── Global State ───────────────────────────────────────────────────────────────
model: Any = None
preprocessor: dict[str, Any] = {}
metadata: dict[str, Any] = {}
feature_columns: list[str] = []


# ─── Lifespan (load model on startup) ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """Load model and preprocessor on startup."""
    global model, preprocessor, metadata, feature_columns

    try:
        model = joblib.load(BEST_MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        metadata = joblib.load(MODEL_METADATA_PATH)
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        print(f"✅ Loaded model: {metadata.get('model_name', 'Unknown')}")
        print(f"   Features: {len(feature_columns)}")
    except FileNotFoundError as e:
        print(f"⚠️  Model files not found: {e}")
        print("   Run the training pipeline first:")
        print("   python data/generate_dataset.py")
        print("   python src/data_engineer.py")
        print("   python src/ml_modeler.py")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

    yield  # App runs here

    # Cleanup (if needed)
    print("🛑 Shutting down API server.")


# ─── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Real-time customer churn probability scoring. "
        "Submit customer features and receive churn risk assessment."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Schemas ────────────────────────────────────────────────────────────────────

class CustomerInput(BaseModel):
    """Input schema for a single customer prediction."""

    gender: Literal["Male", "Female"] = Field(..., description="Customer gender")
    SeniorCitizen: Literal[0, 1] = Field(..., description="1 if senior citizen, 0 otherwise")
    Partner: Literal["Yes", "No"] = Field(..., description="Has a partner")
    Dependents: Literal["Yes", "No"] = Field(..., description="Has dependents")
    tenure: int = Field(..., ge=0, le=72, description="Months with the company")
    PhoneService: Literal["Yes", "No"] = Field(..., description="Has phone service")
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(
        ..., description="Has multiple lines"
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Internet service type"
    )
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Has online security"
    )
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Has online backup"
    )
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Has device protection"
    )
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Has tech support"
    )
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Has streaming TV"
    )
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Has streaming movies"
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Contract type"
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        ..., description="Has paperless billing"
    )
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges in USD")
    TotalCharges: float = Field(..., ge=0, description="Total charges in USD")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Output schema for a churn prediction."""

    churn_probability: float = Field(..., description="Probability of churn (0.0–1.0)")
    churn_prediction: str = Field(..., description="Predicted label: Yes or No")
    risk_level: str = Field(..., description="Risk level: Low, Medium, or High")
    top_risk_factors: list[str] = Field(
        ..., description="Top contributing features to churn risk"
    )


class BatchInput(BaseModel):
    """Input schema for batch predictions."""

    customers: list[CustomerInput] = Field(
        ..., min_length=1, max_length=1000, description="List of customers"
    )


class BatchResponse(BaseModel):
    """Output schema for batch predictions."""

    predictions: list[PredictionResponse]
    total: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_name: str | None
    version: str


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_name: str
    metrics: dict[str, float]
    feature_importances: list[dict[str, Any]]
    total_features: int


# ─── Preprocessing Helper ──────────────────────────────────────────────────────

def preprocess_input(customer: CustomerInput) -> pd.DataFrame:
    """
    Transform a CustomerInput into a feature vector matching the trained model.
    """
    if not preprocessor:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the training pipeline first.",
        )

    data = customer.model_dump()

    # ── Binary encoding ─────────────────────────────────────────────────────
    binary_mapping: dict[str, dict[str, int]] = preprocessor["binary_mapping"]
    for col, mapping in binary_mapping.items():
        data[col] = mapping[data[col]]

    # ── Service-dependent encoding ──────────────────────────────────────────
    service_mapping: dict[str, int] = preprocessor["service_mapping"]
    for col in preprocessor["service_binary_cols"]:
        data[col] = service_mapping[data[col]]

    # ── One-hot encoding ────────────────────────────────────────────────────
    multiclass_data = {col: [data.pop(col)] for col in preprocessor["multiclass_cols"]}
    ohe = preprocessor["ohe"]
    ohe_df = pd.DataFrame(multiclass_data)
    try:
        ohe_result = ohe.transform(ohe_df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Encoding error: {e}")
    ohe_col_names = ohe.get_feature_names_out(preprocessor["multiclass_cols"]).tolist()

    # ── Build feature DataFrame ─────────────────────────────────────────────
    row = {k: [v] for k, v in data.items()}
    df = pd.DataFrame(row)

    # Add one-hot columns
    for i, col_name in enumerate(ohe_col_names):
        df[col_name] = ohe_result[0, i]

    # ── Scale numerical features ────────────────────────────────────────────
    scaler = preprocessor["scaler"]
    num_cols = preprocessor["numerical_cols"]
    df[num_cols] = scaler.transform(df[num_cols])

    # ── Reorder to match training columns ───────────────────────────────────
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df


def classify_risk(probability: float) -> str:
    """Classify churn probability into risk levels."""
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Medium"
    else:
        return "High"


def get_top_risk_factors(
    features: pd.DataFrame,
    n: int = 3,
) -> list[str]:
    """Get top risk factors based on feature importances and input values."""
    if not metadata.get("feature_importances"):
        return []

    importance_map = dict(metadata["feature_importances"])
    # Weight importance by actual feature value magnitude
    scores: dict[str, float] = {}
    for col in features.columns:
        if col in importance_map:
            val = abs(float(features[col].iloc[0]))
            scores[col] = importance_map[col] * (1 + val)

    sorted_factors = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in sorted_factors[:n]]


# ─── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_name=metadata.get("model_name"),
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_churn(customer: CustomerInput) -> PredictionResponse:
    """
    Predict churn probability for a single customer.

    Returns churn probability, prediction label, risk level,
    and top contributing risk factors.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the training pipeline first.",
        )

    try:
        features = preprocess_input(customer)
        probability: float = float(model.predict_proba(features)[:, 1][0])
        prediction: str = "Yes" if probability >= 0.5 else "No"
        risk = classify_risk(probability)
        top_factors = get_top_risk_factors(features)

        return PredictionResponse(
            churn_probability=round(probability, 4),
            churn_prediction=prediction,
            risk_level=risk,
            top_risk_factors=top_factors,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchResponse, tags=["Predictions"])
async def predict_churn_batch(batch: BatchInput) -> BatchResponse:
    """
    Predict churn probability for a batch of customers.

    Accepts up to 1000 customers in a single request.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the training pipeline first.",
        )

    predictions: list[PredictionResponse] = []
    for customer in batch.customers:
        try:
            features = preprocess_input(customer)
            probability = float(model.predict_proba(features)[:, 1][0])
            prediction = "Yes" if probability >= 0.5 else "No"
            risk = classify_risk(probability)
            top_factors = get_top_risk_factors(features)

            predictions.append(
                PredictionResponse(
                    churn_probability=round(probability, 4),
                    churn_prediction=prediction,
                    risk_level=risk,
                    top_risk_factors=top_factors,
                )
            )
        except Exception as e:
            predictions.append(
                PredictionResponse(
                    churn_probability=-1.0,
                    churn_prediction="Error",
                    risk_level="Unknown",
                    top_risk_factors=[str(e)],
                )
            )

    return BatchResponse(predictions=predictions, total=len(predictions))


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info() -> ModelInfoResponse:
    """
    Get model information including metrics and feature importances.
    """
    if not metadata:
        raise HTTPException(
            status_code=503,
            detail="Model metadata not loaded.",
        )

    importances = [
        {"feature": name, "importance": round(val, 4)}
        for name, val in metadata.get("feature_importances", [])
    ]

    return ModelInfoResponse(
        model_name=metadata.get("model_name", "Unknown"),
        metrics=metadata.get("metrics", {}),
        feature_importances=importances,
        total_features=len(feature_columns),
    )
