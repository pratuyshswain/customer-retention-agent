"""
ML Modeling Module — Training, Hyperparameter Tuning, Evaluation.

Handles:
1. Stratified train/test split
2. Logistic Regression baseline
3. Random Forest with GridSearchCV
4. Model comparison and selection
5. Save best model + evaluation report
"""

import os
import sys
import warnings
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH: str = os.path.join(BASE_DIR, "data", "processed_churn.csv")
FEATURE_COLUMNS_PATH: str = os.path.join(BASE_DIR, "models", "feature_columns.joblib")
BEST_MODEL_PATH: str = os.path.join(BASE_DIR, "models", "best_model.joblib")
MODEL_METADATA_PATH: str = os.path.join(BASE_DIR, "models", "model_metadata.joblib")
ARTIFACTS_DIR: str = os.path.join(BASE_DIR, "artifacts")

# ─── Constants ──────────────────────────────────────────────────────────────────
TEST_SIZE: float = 0.20
RANDOM_STATE: int = 42


# ─── 1. Data Loading ───────────────────────────────────────────────────────────

def load_processed_data(
    path: str = PROCESSED_DATA_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load processed data and split into features/target."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"❌ Processed data not found: {path}")
        print("   Run `python src/data_engineer.py` first.")
        sys.exit(1)

    feature_cols: list[str] = joblib.load(FEATURE_COLUMNS_PATH)
    X = df[feature_cols]
    y = df["Churn"]
    print(f"✅ Loaded processed data: {X.shape[0]} rows × {X.shape[1]} features")
    print(f"   Class distribution: {dict(y.value_counts())}")
    return X, y


# ─── 2. Train/Test Split ───────────────────────────────────────────────────────

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split preserving churn ratio."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\n📊 Train/Test Split (stratified, {1-test_size:.0%}/{test_size:.0%}):")
    print(f"   Train: {X_train.shape[0]} rows (churn rate: {y_train.mean():.1%})")
    print(f"   Test:  {X_test.shape[0]} rows (churn rate: {y_test.mean():.1%})")
    return X_train, X_test, y_train, y_test


# ─── 3. Model Evaluation ───────────────────────────────────────────────────────

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict[str, float]:
    """
    Evaluate a trained model on the test set.

    Returns:
        Dictionary of metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    print(f"\n{'─' * 40}")
    print(f"  {model_name} — Test Set Metrics")
    print(f"{'─' * 40}")
    for name, val in metrics.items():
        print(f"   {name:>12}: {val:.4f}")

    return metrics


# ─── 4. Logistic Regression ────────────────────────────────────────────────────

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LogisticRegression:
    """Train a Logistic Regression baseline."""
    print("\n🔧 Training Logistic Regression (baseline)...")
    model = LogisticRegression(
        class_weight="balanced",
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    print("   ✅ Training complete.")
    return model


# ─── 5. Random Forest + GridSearchCV ───────────────────────────────────────────

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """Train a Random Forest with hyperparameter tuning."""
    print("\n🔧 Training Random Forest (with GridSearchCV)...")

    param_grid: dict[str, list[Any]] = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced"],
    }

    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    best_model: RandomForestClassifier = grid_search.best_estimator_
    print(f"   ✅ Training complete.")
    print(f"   Best params: {grid_search.best_params_}")
    print(f"   Best CV ROC-AUC: {grid_search.best_score_:.4f}")

    return best_model


# ─── 6. Feature Importances ────────────────────────────────────────────────────

def get_feature_importances(
    model: Any,
    feature_names: list[str],
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """Extract top feature importances from a model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return []

    indices = np.argsort(importances)[::-1][:top_n]
    return [(feature_names[i], float(importances[i])) for i in indices]


# ─── 7. Visualization ──────────────────────────────────────────────────────────

def plot_model_comparison(
    metrics_lr: dict[str, float],
    metrics_rf: dict[str, float],
    model_lr: Any,
    model_rf: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
) -> None:
    """Generate comparison plots."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── ROC Curves ──────────────────────────────────────────────────────────
    for model, name, color in [
        (model_lr, "Logistic Regression", "#3498db"),
        (model_rf, "Random Forest", "#e74c3c"),
    ]:
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves", fontweight="bold")
    axes[0].legend(loc="lower right")

    # ── Metric Comparison Bar Chart ─────────────────────────────────────────
    metric_names = list(metrics_lr.keys())
    x = np.arange(len(metric_names))
    width = 0.35

    axes[1].bar(x - width / 2, list(metrics_lr.values()), width, label="LR", color="#3498db")
    axes[1].bar(x + width / 2, list(metrics_rf.values()), width, label="RF", color="#e74c3c")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metric_names, rotation=45)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Model Comparison", fontweight="bold")
    axes[1].legend()

    # ── Feature Importances (best model) ────────────────────────────────────
    importances = get_feature_importances(model_rf, feature_names, top_n=10)
    if importances:
        names, vals = zip(*importances)
        y_pos = np.arange(len(names))
        axes[2].barh(y_pos, vals, color="#2ecc71", edgecolor="#27ae60")
        axes[2].set_yticks(y_pos)
        axes[2].set_yticklabels(names)
        axes[2].invert_yaxis()
        axes[2].set_title("Top 10 Feature Importances (RF)", fontweight="bold")

    plt.tight_layout()
    plot_path = os.path.join(ARTIFACTS_DIR, "model_comparison.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n📈 Model comparison plot saved: {plot_path}")


# ─── 8. Report Generation ──────────────────────────────────────────────────────

def generate_report(
    metrics_lr: dict[str, float],
    metrics_rf: dict[str, float],
    best_model_name: str,
    best_params: dict[str, Any] | None,
    feature_importances: list[tuple[str, float]],
) -> str:
    """Generate a markdown evaluation report."""
    report_lines: list[str] = [
        "# Customer Churn Prediction — Model Evaluation Report\n",
        f"## Best Model: **{best_model_name}**\n",
        "---\n",
        "## Metric Comparison\n",
        "| Metric | Logistic Regression | Random Forest |",
        "|--------|:-------------------:|:-------------:|",
    ]

    for metric in metrics_lr:
        lr_val = metrics_lr[metric]
        rf_val = metrics_rf[metric]
        best_marker_lr = " ✅" if lr_val > rf_val else ""
        best_marker_rf = " ✅" if rf_val > lr_val else ""
        report_lines.append(
            f"| {metric.upper()} | {lr_val:.4f}{best_marker_lr} | {rf_val:.4f}{best_marker_rf} |"
        )

    if best_params:
        report_lines.extend([
            "\n## Best Hyperparameters\n",
            "| Parameter | Value |",
            "|-----------|-------|",
        ])
        for k, v in best_params.items():
            report_lines.append(f"| {k} | {v} |")

    if feature_importances:
        report_lines.extend([
            "\n## Top 10 Feature Importances\n",
            "| Rank | Feature | Importance |",
            "|:----:|---------|:----------:|",
        ])
        for i, (name, val) in enumerate(feature_importances, 1):
            report_lines.append(f"| {i} | {name} | {val:.4f} |")

    report_lines.extend([
        "\n## Performance Targets\n",
        "| Metric | Target | Achieved | Status |",
        "|--------|:------:|:--------:|:------:|",
    ])
    best_metrics = metrics_rf if best_model_name == "Random Forest" else metrics_lr
    targets = {"roc_auc": 0.82, "f1": 0.60, "recall": 0.70}
    for metric, target in targets.items():
        achieved = best_metrics.get(metric, 0)
        status = "✅ PASS" if achieved >= target else "❌ FAIL"
        report_lines.append(f"| {metric.upper()} | ≥{target:.2f} | {achieved:.4f} | {status} |")

    return "\n".join(report_lines)


# ─── 9. Full Training Pipeline ─────────────────────────────────────────────────

def run_training_pipeline() -> dict[str, Any]:
    """
    Execute the full ML training pipeline.

    Returns:
        Dictionary with best model info and metrics.
    """
    print("=" * 60)
    print("  ML Modeling Pipeline")
    print("=" * 60)

    # Load data
    X, y = load_processed_data()
    feature_names = X.columns.tolist()

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)
    metrics_lr = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

    # Train Random Forest
    rf_model = train_random_forest(X_train, y_train)
    metrics_rf = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # ── Select best model by ROC-AUC (tiebreak: F1) ────────────────────────
    lr_score = (metrics_lr["roc_auc"], metrics_lr["f1"])
    rf_score = (metrics_rf["roc_auc"], metrics_rf["f1"])

    if rf_score >= lr_score:
        best_model = rf_model
        best_model_name = "Random Forest"
        best_metrics = metrics_rf
        best_params = rf_model.get_params()
    else:
        best_model = lr_model
        best_model_name = "Logistic Regression"
        best_metrics = metrics_lr
        best_params = None

    print(f"\n🏆 Best Model: {best_model_name} (ROC-AUC: {best_metrics['roc_auc']:.4f})")

    # ── Feature importances ─────────────────────────────────────────────────
    importances = get_feature_importances(best_model, feature_names, top_n=10)

    # ── Visualizations ──────────────────────────────────────────────────────
    try:
        plot_model_comparison(
            metrics_lr, metrics_rf, lr_model, rf_model,
            X_test, y_test, feature_names,
        )
    except Exception as e:
        print(f"⚠️  Plot generation failed: {e}")

    # ── Save best model ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"💾 Saved best model: {BEST_MODEL_PATH}")

    # ── Save model metadata ─────────────────────────────────────────────────
    metadata: dict[str, Any] = {
        "model_name": best_model_name,
        "metrics": best_metrics,
        "feature_names": feature_names,
        "feature_importances": importances,
        "best_params": best_params if best_model_name == "Random Forest" else None,
    }
    joblib.dump(metadata, MODEL_METADATA_PATH)
    print(f"💾 Saved model metadata: {MODEL_METADATA_PATH}")

    # ── Generate report ─────────────────────────────────────────────────────
    report = generate_report(
        metrics_lr, metrics_rf, best_model_name,
        best_params if best_model_name == "Random Forest" else None,
        importances,
    )
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    report_path = os.path.join(ARTIFACTS_DIR, "model_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"📄 Saved evaluation report: {report_path}")

    print("\n✅ Training pipeline complete.")
    return metadata


def main() -> None:
    """Run the ML training pipeline."""
    run_training_pipeline()


if __name__ == "__main__":
    main()
