"""
Data Engineering Module — EDA, Cleaning, Feature Engineering.

Handles the full preprocessing pipeline:
1. Load raw CSV
2. Exploratory Data Analysis (correlation heatmap, distributions)
3. Clean missing values (TotalCharges)
4. Encode categorical features
5. Scale numerical features
6. Save processed data + fitted preprocessor
"""

import os
import sys
import warnings
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH: str = os.path.join(BASE_DIR, "data", "telco_churn.csv")
PROCESSED_DATA_PATH: str = os.path.join(BASE_DIR, "data", "processed_churn.csv")
PREPROCESSOR_PATH: str = os.path.join(BASE_DIR, "models", "preprocessor.joblib")
FEATURE_COLUMNS_PATH: str = os.path.join(BASE_DIR, "models", "feature_columns.joblib")
ARTIFACTS_DIR: str = os.path.join(BASE_DIR, "artifacts")


# ─── 1. Data Loading ───────────────────────────────────────────────────────────

def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV."""
    try:
        df = pd.read_csv(path)
        print(f"✅ Loaded raw data: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        print("   Run `python data/generate_dataset.py` first.")
        sys.exit(1)


# ─── 2. Exploratory Data Analysis ──────────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> dict[str, Any]:
    """
    Perform EDA and save visualizations.

    Returns:
        Dictionary with EDA summary statistics.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    stats: dict[str, Any] = {}

    # ── Churn distribution ──────────────────────────────────────────────────
    churn_counts = df["Churn"].value_counts()
    stats["churn_rate"] = churn_counts.get("Yes", 0) / len(df)
    stats["total_rows"] = len(df)
    stats["total_columns"] = len(df.columns)

    print(f"\n📊 EDA Summary:")
    print(f"   Total rows: {stats['total_rows']}")
    print(f"   Churn rate: {stats['churn_rate']:.1%}")

    # ── Missing values ──────────────────────────────────────────────────────
    # TotalCharges has whitespace-only entries in the real (and synthetic) data
    df_temp = df.copy()
    df_temp["TotalCharges"] = pd.to_numeric(df_temp["TotalCharges"], errors="coerce")
    missing = df_temp.isnull().sum()
    stats["missing_values"] = missing[missing > 0].to_dict()
    if stats["missing_values"]:
        print(f"   Missing values: {stats['missing_values']}")

    # ── Correlation heatmap (numeric columns only) ──────────────────────────
    try:
        numeric_df = df_temp.select_dtypes(include=[np.number])
        # Encode Churn as numeric for correlation
        numeric_df = numeric_df.copy()
        numeric_df["Churn"] = (df["Churn"] == "Yes").astype(int)

        fig, ax = plt.subplots(figsize=(10, 8))
        corr = numeric_df.corr()
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            square=True,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
        plt.tight_layout()
        heatmap_path = os.path.join(ARTIFACTS_DIR, "correlation_heatmap.png")
        fig.savefig(heatmap_path, dpi=150)
        plt.close(fig)
        print(f"   📈 Heatmap saved: {heatmap_path}")
    except Exception as e:
        print(f"   ⚠️  Heatmap generation failed: {e}")

    # ── Churn by Contract Type ──────────────────────────────────────────────
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Churn distribution
        churn_counts.plot(kind="bar", ax=axes[0], color=["#2ecc71", "#e74c3c"])
        axes[0].set_title("Churn Distribution", fontweight="bold")
        axes[0].set_ylabel("Count")
        axes[0].tick_params(axis='x', rotation=0)

        # Churn by contract
        contract_churn = pd.crosstab(df["Contract"], df["Churn"], normalize="index")
        contract_churn.plot(kind="bar", stacked=True, ax=axes[1], color=["#2ecc71", "#e74c3c"])
        axes[1].set_title("Churn Rate by Contract", fontweight="bold")
        axes[1].set_ylabel("Proportion")
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend(title="Churn")

        # Tenure distribution by churn
        for label, color in [("No", "#2ecc71"), ("Yes", "#e74c3c")]:
            subset = df[df["Churn"] == label]["tenure"]
            axes[2].hist(subset, bins=30, alpha=0.6, label=label, color=color)
        axes[2].set_title("Tenure by Churn Status", fontweight="bold")
        axes[2].set_xlabel("Tenure (months)")
        axes[2].legend(title="Churn")

        plt.tight_layout()
        eda_path = os.path.join(ARTIFACTS_DIR, "eda_plots.png")
        fig.savefig(eda_path, dpi=150)
        plt.close(fig)
        print(f"   📈 EDA plots saved: {eda_path}")
    except Exception as e:
        print(f"   ⚠️  EDA plots generation failed: {e}")

    return stats


# ─── 3. Data Cleaning ──────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw DataFrame.

    - Convert TotalCharges to numeric (handle whitespace)
    - Impute missing TotalCharges with median
    - Drop customerID
    - Encode target variable (Churn → 0/1)
    """
    df = df.copy()

    # ── TotalCharges: str → float ───────────────────────────────────────────
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    median_tc: float = df["TotalCharges"].median()
    n_missing: int = df["TotalCharges"].isnull().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_tc)
    print(f"\n🧹 Cleaning:")
    print(f"   Imputed {n_missing} missing TotalCharges with median ({median_tc:.2f})")

    # ── Drop customerID ─────────────────────────────────────────────────────
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        print("   Dropped customerID (non-predictive)")

    # ── Encode target ───────────────────────────────────────────────────────
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    print(f"   Encoded Churn: Yes→1, No→0")

    return df


# ─── 4. Feature Engineering ────────────────────────────────────────────────────

def build_preprocessor(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, ColumnTransformer, list[str]]:
    """
    Build and fit the preprocessing pipeline.

    - Binary categoricals → LabelEncoder (Yes/No → 1/0)
    - Multi-class categoricals → OneHotEncoder
    - Numerical → StandardScaler

    Returns:
        (processed_df, fitted_preprocessor, feature_column_names)
    """
    df = df.copy()

    # ── Identify column types ───────────────────────────────────────────────
    binary_cols: list[str] = [
        "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    ]
    # Columns that depend on having a service
    service_binary_cols: list[str] = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    multiclass_cols: list[str] = [
        "InternetService", "Contract", "PaymentMethod",
    ]
    numerical_cols: list[str] = [
        "tenure", "MonthlyCharges", "TotalCharges",
    ]
    passthrough_cols: list[str] = ["SeniorCitizen"]  # Already 0/1

    # ── Encode binary features manually (vectorized) ────────────────────────
    binary_mapping: dict[str, dict[str, int]] = {
        "gender": {"Female": 0, "Male": 1},
        "Partner": {"No": 0, "Yes": 1},
        "Dependents": {"No": 0, "Yes": 1},
        "PhoneService": {"No": 0, "Yes": 1},
        "PaperlessBilling": {"No": 0, "Yes": 1},
    }
    for col, mapping in binary_mapping.items():
        df[col] = df[col].map(mapping).astype(int)

    # ── Encode service-dependent binary features ────────────────────────────
    service_mapping: dict[str, int] = {
        "Yes": 2,
        "No": 1,
        "No phone service": 0,
        "No internet service": 0,
    }
    for col in service_binary_cols:
        df[col] = df[col].map(service_mapping).astype(int)

    # ── One-hot encode multi-class categoricals ─────────────────────────────
    ohe = OneHotEncoder(sparse_output=False, drop="first", dtype=int)
    ohe_features = ohe.fit_transform(df[multiclass_cols])
    ohe_col_names = ohe.get_feature_names_out(multiclass_cols).tolist()
    ohe_df = pd.DataFrame(ohe_features, columns=ohe_col_names, index=df.index)

    # Drop original multi-class columns and concat encoded
    df = df.drop(columns=multiclass_cols)
    df = pd.concat([df, ohe_df], axis=1)

    # ── Scale numerical features ────────────────────────────────────────────
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # ── Build a ColumnTransformer for API-time preprocessing ────────────────
    # We save the scaler and OHE separately for the API to use
    preprocessor_bundle = {
        "binary_mapping": binary_mapping,
        "service_mapping": service_mapping,
        "service_binary_cols": service_binary_cols,
        "multiclass_cols": ["InternetService", "Contract", "PaymentMethod"],
        "numerical_cols": numerical_cols,
        "ohe": ohe,
        "scaler": scaler,
    }

    # Separate features and target
    feature_cols = [c for c in df.columns if c != "Churn"]

    print(f"\n⚙️  Feature Engineering:")
    print(f"   Binary encoded: {len(binary_cols)} columns")
    print(f"   Service encoded: {len(service_binary_cols)} columns")
    print(f"   One-hot encoded: {len(multiclass_cols)} → {len(ohe_col_names)} columns")
    print(f"   Scaled: {len(numerical_cols)} numerical columns")
    print(f"   Final feature count: {len(feature_cols)}")

    return df, preprocessor_bundle, feature_cols


# ─── 5. Full Pipeline ──────────────────────────────────────────────────────────

def run_pipeline(raw_data_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Execute the full data engineering pipeline.

    Returns:
        Processed DataFrame ready for modeling.
    """
    print("=" * 60)
    print("  Data Engineering Pipeline")
    print("=" * 60)

    # Load
    df = load_raw_data(raw_data_path)

    # EDA
    run_eda(df)

    # Clean
    df = clean_data(df)

    # Feature engineering
    df, preprocessor, feature_cols = build_preprocessor(df)

    # Save processed data
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"\n💾 Saved processed data: {PROCESSED_DATA_PATH}")
    print(f"   Shape: {df.shape}")

    # Save preprocessor
    os.makedirs(os.path.dirname(PREPROCESSOR_PATH), exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"💾 Saved preprocessor: {PREPROCESSOR_PATH}")

    # Save feature column names
    joblib.dump(feature_cols, FEATURE_COLUMNS_PATH)
    print(f"💾 Saved feature columns: {FEATURE_COLUMNS_PATH}")

    # Final validation
    assert df.isnull().sum().sum() == 0, "Processed data still contains null values!"
    assert df.select_dtypes(include=["object"]).shape[1] == 0, (
        "Processed data still contains non-numeric columns!"
    )
    print("\n✅ Pipeline complete — all validations passed.")
    return df


def main() -> None:
    """Run the data engineering pipeline."""
    run_pipeline()


if __name__ == "__main__":
    main()
