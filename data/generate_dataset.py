"""
Synthetic Telco Customer Churn Dataset Generator.

Generates a dataset mirroring the Kaggle Telco Customer Churn schema
(21 features, ~7043 rows) with realistic distributions and correlations.
"""

import os
import string
import numpy as np
import pandas as pd
from typing import Final

# ─── Constants ──────────────────────────────────────────────────────────────────
NUM_ROWS: Final[int] = 7043
CHURN_RATE: Final[float] = 0.265
RANDOM_SEED: Final[int] = 42
OUTPUT_DIR: Final[str] = os.path.join(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH: Final[str] = os.path.join(OUTPUT_DIR, "telco_churn.csv")

# ─── Generator ──────────────────────────────────────────────────────────────────

def _generate_customer_ids(n: int, rng: np.random.Generator) -> list[str]:
    """Generate unique customer IDs like '7590-VHVEG'."""
    ids: list[str] = []
    seen: set[str] = set()
    while len(ids) < n:
        prefix = "".join(rng.choice(list(string.digits), size=4))
        suffix = "".join(rng.choice(list(string.ascii_uppercase), size=5))
        cid = f"{prefix}-{suffix}"
        if cid not in seen:
            seen.add(cid)
            ids.append(cid)
    return ids


def _internet_dependent_feature(
    internet_service: pd.Series,
    rng: np.random.Generator,
    yes_prob_dsl: float = 0.40,
    yes_prob_fiber: float = 0.35,
) -> pd.Series:
    """Generate an internet-dependent feature (e.g. OnlineSecurity)."""
    result = pd.Series("No internet service", index=internet_service.index)
    for svc, prob in [("DSL", yes_prob_dsl), ("Fiber optic", yes_prob_fiber)]:
        mask = internet_service == svc
        n_yes = int(mask.sum() * prob)
        indices = internet_service[mask].index.to_numpy().copy()
        rng.shuffle(indices)
        result.loc[indices[:n_yes]] = "Yes"
        result.loc[indices[n_yes:]] = "No"
    return result


def generate_telco_dataset(
    n_rows: int = NUM_ROWS,
    churn_rate: float = CHURN_RATE,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic Telco Customer Churn dataset.

    Args:
        n_rows: Number of rows to generate.
        churn_rate: Target churn rate (fraction).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with 21 columns matching the Telco schema.
    """
    rng = np.random.default_rng(seed)

    # ── Demographics ────────────────────────────────────────────────────────
    gender = rng.choice(["Male", "Female"], size=n_rows)
    senior_citizen = rng.choice([0, 1], size=n_rows, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n_rows, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n_rows, p=[0.30, 0.70])

    # ── Tenure (correlated with churn) ──────────────────────────────────────
    # Churners tend to have shorter tenure
    n_churn = int(n_rows * churn_rate)
    n_no_churn = n_rows - n_churn
    churn_labels = np.array(["Yes"] * n_churn + ["No"] * n_no_churn)
    rng.shuffle(churn_labels)

    tenure = np.zeros(n_rows, dtype=int)
    churn_mask = churn_labels == "Yes"
    # Churners: skewed toward short tenure (exponential-ish)
    tenure[churn_mask] = np.clip(
        rng.exponential(scale=15, size=churn_mask.sum()).astype(int), 0, 72
    )
    # Non-churners: more uniform, slight bias toward longer tenure
    tenure[~churn_mask] = np.clip(
        rng.integers(0, 73, size=(~churn_mask).sum()) , 0, 72
    )

    # ── Services ────────────────────────────────────────────────────────────
    phone_service = rng.choice(["Yes", "No"], size=n_rows, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], size=n_rows, p=[0.42, 0.58]),
    )

    # Internet: fiber optic users churn more
    internet_probs = np.where(
        churn_mask,
        rng.choice(
            ["DSL", "Fiber optic", "No"],
            size=n_rows,
            p=[0.24, 0.56, 0.20],
        ),
        rng.choice(
            ["DSL", "Fiber optic", "No"],
            size=n_rows,
            p=[0.34, 0.31, 0.35],
        ),
    )
    internet_service = pd.Series(internet_probs)

    online_security = _internet_dependent_feature(internet_service, rng, 0.40, 0.17)
    online_backup = _internet_dependent_feature(internet_service, rng, 0.45, 0.28)
    device_protection = _internet_dependent_feature(internet_service, rng, 0.44, 0.29)
    tech_support = _internet_dependent_feature(internet_service, rng, 0.40, 0.17)
    streaming_tv = _internet_dependent_feature(internet_service, rng, 0.48, 0.44)
    streaming_movies = _internet_dependent_feature(internet_service, rng, 0.48, 0.44)

    # ── Contract (strongly correlated with churn) ───────────────────────────
    contract = np.where(
        churn_mask,
        rng.choice(
            ["Month-to-month", "One year", "Two year"],
            size=n_rows,
            p=[0.89, 0.08, 0.03],
        ),
        rng.choice(
            ["Month-to-month", "One year", "Two year"],
            size=n_rows,
            p=[0.43, 0.24, 0.33],
        ),
    )

    # ── Billing ─────────────────────────────────────────────────────────────
    paperless_billing = np.where(
        churn_mask,
        rng.choice(["Yes", "No"], size=n_rows, p=[0.75, 0.25]),
        rng.choice(["Yes", "No"], size=n_rows, p=[0.53, 0.47]),
    )

    payment_methods = [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    payment_method = np.where(
        churn_mask,
        rng.choice(payment_methods, size=n_rows, p=[0.57, 0.17, 0.13, 0.13]),
        rng.choice(payment_methods, size=n_rows, p=[0.22, 0.22, 0.28, 0.28]),
    )

    # ── Charges ─────────────────────────────────────────────────────────────
    base_charge = 18.25
    monthly_charges = base_charge + rng.uniform(0, 100.5, size=n_rows)
    # Fiber optic users pay more
    fiber_mask = internet_service == "Fiber optic"
    monthly_charges[fiber_mask] += rng.uniform(10, 30, size=fiber_mask.sum())
    monthly_charges = np.clip(monthly_charges, 18.25, 118.75).round(2)

    # TotalCharges ~ tenure * MonthlyCharges + noise
    total_charges = (tenure * monthly_charges + rng.normal(0, 50, size=n_rows)).round(2)
    total_charges = np.maximum(total_charges, monthly_charges)  # At least one month

    # ── Construct DataFrame ─────────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "customerID": _generate_customer_ids(n_rows, rng),
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn_labels,
        }
    )

    # ── Introduce ~11 missing TotalCharges (mirroring real dataset) ─────────
    # In the real dataset, new customers (tenure=0) have blank TotalCharges
    zero_tenure_idx = df[df["tenure"] == 0].index
    if len(zero_tenure_idx) >= 11:
        missing_idx = rng.choice(zero_tenure_idx, size=11, replace=False)
    else:
        missing_idx = zero_tenure_idx
    # Convert to string column first (mirrors real CSV quirk), then blank out
    df["TotalCharges"] = df["TotalCharges"].astype(str)
    df.loc[missing_idx, "TotalCharges"] = " "

    return df


def main() -> None:
    """Generate and save the synthetic dataset."""
    print("=" * 60)
    print("  Telco Customer Churn -- Synthetic Dataset Generator")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = generate_telco_dataset()
    df.to_csv(OUTPUT_PATH, index=False)

    # ── Summary stats ───────────────────────────────────────────────────────
    churn_counts = df["Churn"].value_counts()
    print(f"\n[OK] Dataset generated: {OUTPUT_PATH}")
    print(f"   Shape: {df.shape}")
    print(f"   Churn distribution:")
    print(f"     No:  {churn_counts.get('No', 0)} ({churn_counts.get('No', 0) / len(df) * 100:.1f}%)")
    print(f"     Yes: {churn_counts.get('Yes', 0)} ({churn_counts.get('Yes', 0) / len(df) * 100:.1f}%)")
    missing_tc = (df["TotalCharges"].str.strip() == "").sum()
    print(f"   Missing TotalCharges: {missing_tc}")
    print()


if __name__ == "__main__":
    main()
