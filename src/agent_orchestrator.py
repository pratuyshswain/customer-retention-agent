"""
Customer Retention Reasoning Orchestrator — ReAct Agent.

Implements a three-layer Thought → Action → Observation architecture:
  Layer 1 (Thought)  — LLM Pre-parsing: Intercepts plain text terminal input,
                        uses gpt-4o-mini to extract structured customer attributes
                        with intelligent defaults for omitted fields.
  Layer 2 (Action)   — Local Tool Invocation: Dynamically loads preprocessor.joblib,
                        feature_columns.joblib, and best_model.joblib from models/,
                        transforms the LLM JSON into the exact binary/one-hot encoded
                        feature vector, and calls predict_proba() for churn risk.
  Layer 3 (Observe)  — Strategy Generation: Feeds the user's original context,
                        extracted structured data, and churn risk score back to
                        gpt-4o-mini for a customized executive retention strategy.

CPU-Optimized: All pandas manipulation and model inference strictly targets
  13th Gen Intel i5 / Iris Xe integrated graphics. No CUDA dependencies.

Microsoft Agents League Hackathon | GIET Baniatangi / BPUT
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ─── Environment & Logging ──────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ChurnAgent")

# ─── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEST_MODEL_PATH: str = os.path.join(BASE_DIR, "models", "best_model.joblib")
PREPROCESSOR_PATH: str = os.path.join(BASE_DIR, "models", "preprocessor.joblib")
MODEL_METADATA_PATH: str = os.path.join(BASE_DIR, "models", "model_metadata.joblib")
FEATURE_COLUMNS_PATH: str = os.path.join(BASE_DIR, "models", "feature_columns.joblib")

# ─── Agent Constants ────────────────────────────────────────────────────────────
LLM_MODEL: str = "gpt-4o-mini"
LLM_TEMPERATURE_EXTRACT: float = 0.1   # Low temperature for deterministic extraction
LLM_TEMPERATURE_STRATEGY: float = 0.7  # Higher creativity for retention strategies
LLM_MAX_TOKENS_EXTRACT: int = 600
LLM_MAX_TOKENS_STRATEGY: int = 1200


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LLM CLIENT INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def _init_openai_client() -> Any:
    """
    Securely initialize the OpenAI-compatible client for Microsoft GitHub Models.

    Uses the free Azure inference endpoint with a GitHub personal access token.

    Raises:
        RuntimeError: If no valid GITHUB_TOKEN is found or openai package is missing.
    """
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token or not github_token.strip() or github_token == "ghp_your_token_here":
        raise RuntimeError(
            "GITHUB_TOKEN not found or not configured.\n"
            "  → Copy .env.example to .env and set your GitHub token.\n"
            "  → Generate a token from: https://github.com/settings/tokens"
        )
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=github_token,
        )
        logger.info("GitHub Models client initialized successfully (model: %s)", LLM_MODEL)
        return client
    except ImportError as exc:
        raise RuntimeError(
            "openai package not installed. Run: pip install openai>=1.30"
        ) from exc


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOCAL ML PIPELINE (CPU-Optimized)
# ═══════════════════════════════════════════════════════════════════════════════

class LocalMLPipeline:
    """
    Manages the trained ML model and preprocessing artifacts.

    Loads preprocessor.joblib, feature_columns.joblib, and best_model.joblib
    from the local models/ directory for CPU-based inference on Intel i5 / Iris Xe.
    """

    def __init__(self) -> None:
        self.model: Any = None
        self.preprocessor: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        self.feature_columns: list[str] = []
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load all serialized ML artifacts from disk."""
        try:
            self.model = joblib.load(BEST_MODEL_PATH)
            self.preprocessor = joblib.load(PREPROCESSOR_PATH)
            self.metadata = joblib.load(MODEL_METADATA_PATH)
            self.feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
            model_name = self.metadata.get("model_name", "Unknown")
            roc_auc = self.metadata.get("metrics", {}).get("roc_auc", 0.0)
            logger.info(
                "ML Pipeline loaded: %s (ROC-AUC: %.4f) — CPU inference ready",
                model_name,
                roc_auc,
            )
        except FileNotFoundError as exc:
            logger.error("Model artifacts not found: %s", exc)
            raise RuntimeError(
                "ML model not available. Run the training pipeline first:\n"
                "  python data/generate_dataset.py && "
                "python src/data_engineer.py && "
                "python src/ml_modeler.py"
            ) from exc

    def preprocess_customer(self, customer_data: dict[str, Any]) -> pd.DataFrame:
        """
        Transform a raw customer attribute dictionary into the exact
        binary/one-hot encoded feature vector the trained model expects.

        This function replicates the encoding logic from data_engineer.py
        using the fitted preprocessor artifacts (OHE, Scaler, mappings).

        Args:
            customer_data: Dictionary of customer attributes matching the
                           schema produced by the LLM extraction layer.

        Returns:
            A single-row DataFrame with columns matching feature_columns.joblib.
        """
        try:
            data = dict(customer_data)  # Shallow copy to avoid mutation

            # Remove non-feature fields
            data.pop("customer_id", None)

            # ── Binary encoding (gender, Partner, Dependents, etc.) ─────────
            for col, mapping in self.preprocessor["binary_mapping"].items():
                if col in data:
                    raw_value = data[col]
                    if raw_value in mapping:
                        data[col] = mapping[raw_value]
                    else:
                        logger.warning(
                            "Unknown value '%s' for binary col '%s', defaulting to 0",
                            raw_value, col,
                        )
                        data[col] = 0

            # ── Service-dependent encoding (MultipleLines, OnlineSecurity, etc.)
            service_mapping = self.preprocessor["service_mapping"]
            for col in self.preprocessor["service_binary_cols"]:
                if col in data:
                    raw_value = data[col]
                    if raw_value in service_mapping:
                        data[col] = service_mapping[raw_value]
                    else:
                        logger.warning(
                            "Unknown value '%s' for service col '%s', defaulting to 1",
                            raw_value, col,
                        )
                        data[col] = 1  # Default to "No" (service exists but not subscribed)

            # ── One-hot encoding (InternetService, Contract, PaymentMethod) ──
            multiclass_cols = self.preprocessor["multiclass_cols"]
            multiclass_data = {}
            for col in multiclass_cols:
                multiclass_data[col] = [data.pop(col)]

            ohe = self.preprocessor["ohe"]
            ohe_result = ohe.transform(pd.DataFrame(multiclass_data))
            ohe_col_names = ohe.get_feature_names_out(multiclass_cols).tolist()

            # ── Build single-row DataFrame ──────────────────────────────────
            row = {k: [v] for k, v in data.items()}
            df = pd.DataFrame(row)

            # Append one-hot encoded columns
            for idx, col_name in enumerate(ohe_col_names):
                df[col_name] = ohe_result[0, idx]

            # ── Scale numerical features ────────────────────────────────────
            scaler = self.preprocessor["scaler"]
            num_cols = self.preprocessor["numerical_cols"]
            df[num_cols] = scaler.transform(df[num_cols])

            # ── Reorder columns to match training order ─────────────────────
            df = df.reindex(columns=self.feature_columns, fill_value=0)

            logger.info(
                "Preprocessing complete: %d features generated",
                len(self.feature_columns),
            )
            return df

        except Exception as exc:
            logger.error("Preprocessing failed: %s", exc)
            raise

    def predict_churn(self, features_df: pd.DataFrame) -> dict[str, Any]:
        """
        Run predict_proba() on the preprocessed feature vector.

        Returns:
            Dictionary with churn_probability, risk_level, top_risk_factors,
            model_name, and model_confidence.
        """
        try:
            # ── Churn probability (CPU inference) ───────────────────────────
            probability = float(self.model.predict_proba(features_df)[:, 1][0])

            # ── Risk classification ─────────────────────────────────────────
            if probability < 0.3:
                risk_level = "Low"
            elif probability < 0.5:
                risk_level = "Medium"
            elif probability < 0.75:
                risk_level = "High"
            else:
                risk_level = "Critical"

            # ── Top risk factors (weighted by feature importance × value) ───
            importances = dict(self.metadata.get("feature_importances", []))
            factor_scores: dict[str, float] = {}
            for col in features_df.columns:
                if col in importances:
                    val = abs(float(features_df[col].iloc[0]))
                    factor_scores[col] = importances[col] * (1 + val)

            sorted_factors = sorted(
                factor_scores.items(), key=lambda x: x[1], reverse=True
            )
            top_factors = [
                name.replace("_", " ").replace("  ", " ")
                for name, _ in sorted_factors[:5]
            ]

            # ── Model confidence from ROC-AUC ──────────────────────────────
            roc_auc = self.metadata.get("metrics", {}).get("roc_auc", 0.0)
            if roc_auc >= 0.85:
                confidence = "High"
            elif roc_auc >= 0.75:
                confidence = "Medium"
            else:
                confidence = "Low"

            result = {
                "churn_probability": round(probability, 4),
                "churn_percentage": f"{probability * 100:.1f}%",
                "risk_level": risk_level,
                "top_risk_factors": top_factors,
                "model_name": self.metadata.get("model_name", "Unknown"),
                "model_confidence": confidence,
                "roc_auc": round(roc_auc, 4),
            }

            logger.info(
                "Prediction complete: %.1f%% churn risk (%s) — Model: %s",
                probability * 100,
                risk_level,
                result["model_name"],
            )
            return result

        except Exception as exc:
            logger.error("Prediction failed: %s", exc)
            raise


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LAYER 1: LLM PRE-PARSING (Thought)
# ═══════════════════════════════════════════════════════════════════════════════

_NLP_EXTRACTION_SYSTEM_PROMPT: str = """\
You are a data-extraction assistant for a telecom customer churn prediction system.

Given a natural-language description of a customer scenario, extract a JSON object
with EXACTLY these fields (use the EXACT key names shown):

  gender            : "Male" or "Female"
  SeniorCitizen     : 0 or 1
  Partner           : "Yes" or "No"
  Dependents        : "Yes" or "No"
  tenure            : integer 0-72 (months with the company)
  PhoneService      : "Yes" or "No"
  MultipleLines     : "Yes", "No", or "No phone service"
  InternetService   : "DSL", "Fiber optic", or "No"
  OnlineSecurity    : "Yes", "No", or "No internet service"
  OnlineBackup      : "Yes", "No", or "No internet service"
  DeviceProtection  : "Yes", "No", or "No internet service"
  TechSupport       : "Yes", "No", or "No internet service"
  StreamingTV       : "Yes", "No", or "No internet service"
  StreamingMovies   : "Yes", "No", or "No internet service"
  Contract          : "Month-to-month", "One year", or "Two year"
  PaperlessBilling  : "Yes" or "No"
  PaymentMethod     : "Electronic check", "Mailed check",
                      "Bank transfer (automatic)", or "Credit card (automatic)"
  MonthlyCharges    : float >= 0 (USD)
  TotalCharges      : float >= 0 (USD)

RULES:
1. Extract any values explicitly stated or clearly implied by the scenario.
2. For fields NOT mentioned, infer REASONABLE DEFAULTS using these heuristics:
   - A short-tenure customer likely has a "Month-to-month" contract.
   - If MonthlyCharges is given but TotalCharges is not, compute:
     TotalCharges = MonthlyCharges × tenure.
   - A customer described as "elderly", "senior", "retired" → SeniorCitizen = 1.
   - If no internet type is specified but internet is implied, assume "DSL" (standard).
   - If internet service is "No", all internet-dependent services must be "No internet service".
   - If PhoneService is "No", MultipleLines must be "No phone service".
   - Default gender to "Male" if not mentioned.
   - Default PhoneService to "Yes", MultipleLines to "No".
   - Default PaperlessBilling to "Yes" for newer/younger customers.
   - Default PaymentMethod to "Electronic check" for short-tenure customers,
     "Bank transfer (automatic)" for long-tenure customers.
   - If no streaming is mentioned, default StreamingTV and StreamingMovies to "No".
   - Default OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport to "No"
     unless explicitly stated.
3. Return ONLY the raw JSON object. No markdown fences, no explanation, no commentary.
4. Every field is REQUIRED — never omit a field.
"""


def extract_customer_features(
    user_text: str,
    llm_client: Any,
) -> dict[str, Any]:
    """
    Layer 1 — Thought: Use gpt-4o-mini to parse natural language into
    a structured customer attribute dictionary.

    Args:
        user_text: The user's plain text input from the terminal.
        llm_client: An initialized OpenAI client instance.

    Returns:
        Dictionary of customer attributes matching the model schema.

    Raises:
        ValueError: If the LLM response cannot be parsed into valid JSON.
    """
    try:
        logger.info("Layer 1 (Thought): Sending text to gpt-4o-mini for feature extraction...")

        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _NLP_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            temperature=LLM_TEMPERATURE_EXTRACT,
            max_tokens=LLM_MAX_TOKENS_EXTRACT,
        )

        raw_content = response.choices[0].message.content.strip()

        # ── Strip markdown fences if the LLM included them ──────────────
        if raw_content.startswith("```"):
            # Remove opening fence (possibly ```json)
            raw_content = raw_content.split("\n", 1)[1] if "\n" in raw_content else raw_content[3:]
        if raw_content.endswith("```"):
            raw_content = raw_content[:-3]
        raw_content = raw_content.strip()

        # ── Parse JSON ──────────────────────────────────────────────────
        customer_data = json.loads(raw_content)

        # ── Validate all required fields are present ────────────────────
        required_fields = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges",
        ]

        missing_fields = [f for f in required_fields if f not in customer_data]
        if missing_fields:
            raise ValueError(
                f"LLM extraction missing required fields: {missing_fields}"
            )

        # ── Assign a dynamic customer ID ────────────────────────────────
        customer_data["customer_id"] = f"NLP-{str(uuid.uuid4())[:6].upper()}"

        # ── Ensure numeric types ────────────────────────────────────────
        customer_data["tenure"] = int(customer_data["tenure"])
        customer_data["SeniorCitizen"] = int(customer_data["SeniorCitizen"])
        customer_data["MonthlyCharges"] = float(customer_data["MonthlyCharges"])
        customer_data["TotalCharges"] = float(customer_data["TotalCharges"])

        logger.info(
            "Layer 1 complete: Extracted profile %s (tenure=%dmo, $%.2f/mo, %s contract)",
            customer_data["customer_id"],
            customer_data["tenure"],
            customer_data["MonthlyCharges"],
            customer_data["Contract"],
        )
        return customer_data

    except json.JSONDecodeError as exc:
        logger.error("Layer 1 failed — LLM returned invalid JSON: %s", exc)
        raise ValueError(
            f"Could not parse LLM response as JSON: {exc}"
        ) from exc
    except Exception as exc:
        logger.error("Layer 1 failed — Feature extraction error: %s", exc)
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — LAYER 3: STRATEGY GENERATION (Observation & Final Answer)
# ═══════════════════════════════════════════════════════════════════════════════

_STRATEGY_SYSTEM_PROMPT: str = """\
You are an elite customer retention strategist for a major telecom company.

You will receive:
1. The customer's ORIGINAL description (how they were described in natural language).
2. The STRUCTURED DATA extracted from that description (JSON).
3. The ML MODEL'S CHURN RISK SCORE and risk factors.

Your job is to produce a HIGHLY CUSTOMIZED, EXECUTIVE-GRADE retention strategy
formatted beautifully for a terminal display.

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🎯  RETENTION STRATEGY REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  📊  RISK ASSESSMENT
  ──────────────────────────────────────────────────────────
  [Display the churn probability, risk level, and model confidence clearly]

  🔍  KEY RISK FACTORS
  ──────────────────────────────────────────────────────────
  [List the top risk factors identified by the ML model and explain WHY
   each contributes to churn risk for this specific customer]

  💡  PERSONALIZED RETENTION STRATEGY
  ──────────────────────────────────────────────────────────
  [Strategy Title — creative and specific to this customer]

  [Detailed, personalized retention offer with specific dollar amounts,
   discount percentages, and service upgrades. Reference the customer's
   actual situation, contract type, payment method, and services.]

  📋  RECOMMENDED ACTIONS
  ──────────────────────────────────────────────────────────
  [Numbered list of 4-6 concrete, actionable steps with timelines]

  ⏰  URGENCY & TIMELINE
  ──────────────────────────────────────────────────────────
  [Specify urgency level and explain why this timeline is critical]

  📈  EXPECTED IMPACT
  ──────────────────────────────────────────────────────────
  [Estimated retention probability improvement with justification]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULES:
1. Be SPECIFIC to this customer — reference their exact charges, tenure, services.
2. Include specific dollar amounts for discounts and savings.
3. Tailor the strategy to their risk level: Critical → aggressive saves,
   Low → engagement and appreciation.
4. Keep the tone professional but empathetic.
5. Maximum discount: 25% on monthly charges.
6. Maximum free months: 3.
7. Always consider the customer's payment method and contract type.
8. If they have technical issues or no security/backup, address those specifically.
"""


def generate_retention_strategy(
    original_text: str,
    customer_data: dict[str, Any],
    prediction: dict[str, Any],
    llm_client: Any,
) -> str:
    """
    Layer 3 — Observation & Final Answer: Feed the full context back to
    gpt-4o-mini to generate a customized executive retention strategy.

    Args:
        original_text: The user's original natural language input.
        customer_data: The structured attributes extracted by Layer 1.
        prediction: The churn prediction results from Layer 2.
        llm_client: An initialized OpenAI client instance.

    Returns:
        Formatted retention strategy string for terminal display.
    """
    try:
        logger.info("Layer 3 (Observation): Generating personalized retention strategy...")

        # ── Build the context message for the LLM ──────────────────────
        context_message = (
            f"ORIGINAL CUSTOMER DESCRIPTION:\n"
            f'"{original_text}"\n\n'
            f"EXTRACTED STRUCTURED DATA:\n"
            f"{json.dumps(customer_data, indent=2)}\n\n"
            f"ML MODEL CHURN PREDICTION:\n"
            f"  • Churn Probability: {prediction['churn_percentage']}\n"
            f"  • Risk Level: {prediction['risk_level']}\n"
            f"  • Model: {prediction['model_name']} "
            f"(Confidence: {prediction['model_confidence']}, "
            f"ROC-AUC: {prediction['roc_auc']})\n"
            f"  • Top Risk Factors: {', '.join(prediction['top_risk_factors'])}\n"
        )

        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _STRATEGY_SYSTEM_PROMPT},
                {"role": "user", "content": context_message},
            ],
            temperature=LLM_TEMPERATURE_STRATEGY,
            max_tokens=LLM_MAX_TOKENS_STRATEGY,
        )

        strategy_text = response.choices[0].message.content.strip()
        logger.info("Layer 3 complete: Retention strategy generated successfully")
        return strategy_text

    except Exception as exc:
        logger.error("Layer 3 failed — Strategy generation error: %s", exc)
        # Return a structured fallback so the user still gets value
        return _build_fallback_strategy(customer_data, prediction)


def _build_fallback_strategy(
    customer_data: dict[str, Any],
    prediction: dict[str, Any],
) -> str:
    """
    Generate a deterministic fallback strategy if the LLM strategy call fails.
    Ensures the user always receives actionable output.
    """
    risk = prediction["risk_level"]
    prob = prediction["churn_percentage"]
    monthly = customer_data.get("MonthlyCharges", 0)
    tenure = customer_data.get("tenure", 0)
    contract = customer_data.get("Contract", "Unknown")
    cid = customer_data.get("customer_id", "Unknown")

    # ── Determine urgency and discount by risk level ────────────────────
    if risk == "Critical":
        urgency = "Immediate"
        discount_pct = 25
        lift = "30-45%"
    elif risk == "High":
        urgency = "Within 48 hours"
        discount_pct = 15
        lift = "20-30%"
    elif risk == "Medium":
        urgency = "This week"
        discount_pct = 10
        lift = "10-20%"
    else:
        urgency = "This month"
        discount_pct = 5
        lift = "5-10%"

    new_rate = monthly * (1 - discount_pct / 100)
    savings = monthly * (discount_pct / 100) * 12

    return (
        f"\n{'━' * 60}\n"
        f"  🎯  RETENTION STRATEGY REPORT (Fallback)\n"
        f"{'━' * 60}\n\n"
        f"  📊  RISK ASSESSMENT\n"
        f"  {'─' * 56}\n"
        f"  Customer ID       : {cid}\n"
        f"  Churn Probability : {prob}\n"
        f"  Risk Level        : {risk}\n"
        f"  Tenure            : {tenure} months\n"
        f"  Contract          : {contract}\n"
        f"  Monthly Charges   : ${monthly:.2f}\n\n"
        f"  🔍  TOP RISK FACTORS\n"
        f"  {'─' * 56}\n"
        f"  {', '.join(prediction.get('top_risk_factors', ['N/A']))}\n\n"
        f"  💡  PERSONALIZED OFFER\n"
        f"  {'─' * 56}\n"
        f"  Apply a {discount_pct}% loyalty discount, reducing monthly charges\n"
        f"  from ${monthly:.2f} to ${new_rate:.2f}/month.\n"
        f"  Estimated annual savings: ${savings:.2f}.\n\n"
        f"  📋  RECOMMENDED ACTIONS\n"
        f"  {'─' * 56}\n"
        f"  1. Contact customer within {urgency.lower()} timeframe\n"
        f"  2. Apply {discount_pct}% loyalty discount upon contract commitment\n"
        f"  3. Schedule follow-up call within 7 days\n"
        f"  4. Enroll in proactive retention monitoring\n\n"
        f"  ⏰  Urgency: {urgency}\n"
        f"  📈  Expected Retention Lift: {lift}\n"
        f"{'━' * 60}\n"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ORCHESTRATOR: THE THREE-LAYER PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class AgentOrchestrator:
    """
    Bridges generative AI reasoning and deterministic ML artifacts.

    Orchestrates the three-layer pipeline:
      1. LLM Pre-parsing (Thought) — Extract features from natural language
      2. Local ML Inference (Action) — Preprocess + predict_proba()
      3. Strategy Generation (Observation) — Generate retention strategy

    CPU-optimized for 13th Gen Intel i5 / Iris Xe integrated graphics.
    """

    def __init__(self) -> None:
        """Initialize the orchestrator with LLM client and ML pipeline."""
        logger.info("Initializing Agent Orchestrator...")

        # ── Initialize OpenAI client (required — no silent fallback) ────
        self.llm_client = _init_openai_client()

        # ── Initialize local ML pipeline ────────────────────────────────
        self.ml_pipeline = LocalMLPipeline()

        logger.info("Agent Orchestrator ready — all systems operational")

    def analyze(self, user_text: str) -> str:
        """
        Run the full three-layer analysis pipeline on a user's text input.

        Args:
            user_text: Plain text description of a customer from the terminal.

        Returns:
            Formatted retention strategy string for terminal display.
        """
        analysis_id = str(uuid.uuid4())[:8].upper()
        logger.info("Starting analysis %s for input: '%s'", analysis_id, user_text[:80])

        # ════════════════════════════════════════════════════════════════
        # LAYER 1 — THOUGHT: LLM Pre-parsing
        # ════════════════════════════════════════════════════════════════
        print(f"\n  {'═' * 58}")
        print(f"  🧠  LAYER 1 — THOUGHT: Parsing Natural Language Input")
        print(f"  {'═' * 58}")
        print(f"  📝  \"{user_text}\"\n")

        try:
            customer_data = extract_customer_features(user_text, self.llm_client)
        except (ValueError, Exception) as exc:
            error_msg = (
                f"\n  ❌  Feature extraction failed: {exc}\n"
                f"  💡  Try rephrasing your description with more specific details.\n"
                f"      Example: \"A 45-year-old male, 3 months tenure, $85/month,\n"
                f"      fiber optic internet, month-to-month contract.\"\n"
            )
            return error_msg

        # ── Display extracted profile ───────────────────────────────────
        print(f"  ✅  Profile extracted → {customer_data['customer_id']}")
        print(f"  {'─' * 56}")
        print(f"  │  Gender          : {customer_data['gender']}")
        print(f"  │  Senior Citizen  : {'Yes' if customer_data['SeniorCitizen'] else 'No'}")
        print(f"  │  Partner         : {customer_data['Partner']}")
        print(f"  │  Dependents      : {customer_data['Dependents']}")
        print(f"  │  Tenure          : {customer_data['tenure']} months")
        print(f"  │  Contract        : {customer_data['Contract']}")
        print(f"  │  Monthly Charges : ${customer_data['MonthlyCharges']:.2f}")
        print(f"  │  Total Charges   : ${customer_data['TotalCharges']:.2f}")
        print(f"  │  Internet        : {customer_data['InternetService']}")
        print(f"  │  Phone Service   : {customer_data['PhoneService']}")
        print(f"  │  Payment Method  : {customer_data['PaymentMethod']}")
        print(f"  │  Online Security : {customer_data['OnlineSecurity']}")
        print(f"  │  Tech Support    : {customer_data['TechSupport']}")
        print(f"  │  Streaming TV    : {customer_data['StreamingTV']}")
        print(f"  │  Streaming Movies: {customer_data['StreamingMovies']}")
        print(f"  {'─' * 56}")

        # ════════════════════════════════════════════════════════════════
        # LAYER 2 — ACTION: Local ML Inference (CPU)
        # ════════════════════════════════════════════════════════════════
        print(f"\n  {'═' * 58}")
        print(f"  ⚡  LAYER 2 — ACTION: Local ML Model Inference (CPU)")
        print(f"  {'═' * 58}")

        try:
            # Preprocess: JSON → feature vector
            features_df = self.ml_pipeline.preprocess_customer(customer_data)
            print(f"  ✅  Feature vector: {len(self.ml_pipeline.feature_columns)} dimensions")

            # Predict: feature vector → churn probability
            prediction = self.ml_pipeline.predict_churn(features_df)

            print(f"  {'─' * 56}")
            print(f"  │  Churn Probability : {prediction['churn_percentage']}")
            print(f"  │  Risk Level        : {prediction['risk_level']}")
            print(f"  │  Model             : {prediction['model_name']}")
            print(f"  │  Model Confidence  : {prediction['model_confidence']} (ROC-AUC: {prediction['roc_auc']})")
            print(f"  │  Top Risk Factors  :")
            for factor in prediction["top_risk_factors"]:
                print(f"  │    • {factor}")
            print(f"  {'─' * 56}")

        except Exception as exc:
            error_msg = (
                f"\n  ❌  ML prediction failed: {exc}\n"
                f"  💡  Ensure model artifacts exist in models/ directory.\n"
            )
            return error_msg

        # ════════════════════════════════════════════════════════════════
        # LAYER 3 — OBSERVATION: Strategy Generation via LLM
        # ════════════════════════════════════════════════════════════════
        print(f"\n  {'═' * 58}")
        print(f"  🔍  LAYER 3 — OBSERVATION: Generating Retention Strategy")
        print(f"  {'═' * 58}")

        strategy = generate_retention_strategy(
            original_text=user_text,
            customer_data=customer_data,
            prediction=prediction,
            llm_client=self.llm_client,
        )

        return strategy


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — INTERACTIVE CLI LOOP
# ═══════════════════════════════════════════════════════════════════════════════

_CLI_BANNER: str = r"""
╔══════════════════════════════════════════════════════════════╗
║   🤖  FOUNDRY CUSTOMER RETENTION REASONING AGENT           ║
║   Microsoft Agents League Hackathon | GIET Baniatangi       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║   Reasoning Engine : Thought → Action → Observation          ║
║   LLM              : {llm_model:<37s}║
║   ML Model         : {ml_model:<37s}║
║   ROC-AUC          : {roc_auc:<37s}║
║   Hardware          : CPU-Optimized (Intel i5 / Iris Xe)     ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║   USAGE                                                      ║
║   • Describe any customer scenario in plain English.         ║
║     e.g. "A senior citizen with 3 months tenure paying $100" ║
║   • The agent will extract features, run the ML model,       ║
║     and generate a personalized retention strategy.          ║
║   • Type 'exit', 'quit', or 'q' to quit.                    ║
╚══════════════════════════════════════════════════════════════╝
"""


def interactive_cli() -> None:
    """
    Launch the interactive terminal CLI for the Foundry Reasoning Agent.

    Implements an infinite input loop that:
      1. Accepts plain English customer descriptions
      2. Routes through the three-layer Thought → Action → Observation pipeline
      3. Displays a customized executive retention strategy
      4. Loops until the user types 'exit', 'quit', or 'q'
    """
    # ── Boot the orchestrator ───────────────────────────────────────────
    try:
        orchestrator = AgentOrchestrator()
    except RuntimeError as exc:
        print(f"\n  ❌  STARTUP FAILED: {exc}\n")
        logger.error("Failed to initialize orchestrator: %s", exc)
        sys.exit(1)

    # ── Display welcome banner ──────────────────────────────────────────
    ml_model_name = orchestrator.ml_pipeline.metadata.get("model_name", "N/A")
    roc_auc = orchestrator.ml_pipeline.metadata.get("metrics", {}).get("roc_auc", 0.0)

    print(_CLI_BANNER.format(
        llm_model=LLM_MODEL,
        ml_model=ml_model_name,
        roc_auc=f"{roc_auc:.4f}",
    ))

    # ── Main interaction loop ───────────────────────────────────────────
    while True:
        try:
            user_input = input("  🗣️  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  👋  Session terminated. Goodbye!\n")
            break

        # Skip empty input
        if not user_input:
            continue

        # ── Exit commands ───────────────────────────────────────────────
        if user_input.lower() in ("exit", "quit", "q"):
            print("\n  👋  Agent shutting down. Goodbye!\n")
            break

        # ── Run the three-layer analysis pipeline ───────────────────────
        try:
            result = orchestrator.analyze(user_input)
            print(result)
        except Exception as exc:
            logger.error("Unexpected error during analysis: %s", exc)
            print(f"\n  ❌  An unexpected error occurred: {exc}")
            print(f"  💡  Please try again with a different description.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — BACKWARD-COMPATIBLE EXPORTS & ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

# ── Re-export classes used by tests and other modules ───────────────────────
# These maintain backward compatibility with test_agent_orchestrator.py
# which imports these symbols directly.

from pydantic import BaseModel, Field
from typing import Literal
from enum import Enum


class RiskLevel(str, Enum):
    """Churn risk classification."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class CustomerProfile(BaseModel):
    """Input schema for a customer to evaluate."""
    customer_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=72)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class RiskPrediction(BaseModel):
    """Output from the Risk Predictor tool."""
    churn_probability: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    top_risk_factors: list[str]
    model_name: str
    model_confidence: str


class SupportTicket(BaseModel):
    """A simulated customer support ticket."""
    ticket_id: str
    date: str
    category: str
    severity: Literal["Low", "Medium", "High", "Critical"]
    resolved: bool
    summary: str


class CustomerContext(BaseModel):
    """Output from the Context Retriever tool."""
    customer_id: str
    account_status: str
    loyalty_tier: str
    lifetime_value: float
    last_interaction_days: int
    recent_tickets: list[SupportTicket]
    contract_renewal_date: str
    payment_issues: int
    nps_score: int | None
    notes: list[str]


class CorporatePolicy(BaseModel):
    """Corporate retention policy constraints for proposal validation."""
    max_discount_pct: float = Field(
        default=25.0,
        description="Maximum allowable discount percentage on monthly charges",
    )
    max_free_months: int = Field(
        default=3,
        description="Maximum number of free months that can be offered",
    )
    eligible_upgrades: list[str] = Field(
        default=["speed_tier", "streaming_bundle", "device_protection", "tech_support"],
        description="Service upgrades available for retention offers",
    )
    requires_contract_commitment: bool = Field(
        default=True,
        description="Whether discounts require a minimum contract commitment",
    )
    min_contract_months: int = Field(
        default=12,
        description="Minimum contract commitment required for discount eligibility",
    )
    max_referral_credit: float = Field(
        default=25.0,
        description="Maximum referral credit in USD per successful referral",
    )
    max_fee_waiver_amount: float = Field(
        default=150.0,
        description="Maximum total fee waiver amount in USD",
    )


class ReasoningStep(BaseModel):
    """A single step in the ReAct reasoning loop."""
    step_number: int
    phase: Literal["Thought", "Action", "Observation"]
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class RetentionProposal(BaseModel):
    """Validated retention proposal from Tool 3."""
    strategy_title: str
    personalized_offer: str
    recommended_actions: list[str]
    estimated_retention_lift: str
    urgency: Literal["Immediate", "Within 48 hours", "This week", "This month"]
    policy_compliant: bool = Field(
        description="Whether the proposal passes all corporate policy checks",
    )
    policy_notes: list[str] = Field(
        default_factory=list,
        description="Policy compliance notes and any adjustments made",
    )


class RetentionStrategy(BaseModel):
    """The final retention offer generated by the agent."""
    customer_id: str
    risk_level: RiskLevel
    churn_probability: float
    strategy_title: str
    personalized_offer: str
    reasoning_summary: str
    recommended_actions: list[str]
    estimated_retention_lift: str
    urgency: Literal["Immediate", "Within 48 hours", "This week", "This month"]
    reasoning_trace: list[ReasoningStep]


# ── Backward-compatible tool classes (used by tests) ────────────────────────

class RiskPredictor:
    """
    Wraps the trained ML model as a callable tool.

    Loads the serialized model, preprocessor, and metadata from disk
    and exposes a predict() method that returns structured risk scores.
    """

    def __init__(self) -> None:
        self._pipeline = LocalMLPipeline()
        self.model = self._pipeline.model
        self.preprocessor = self._pipeline.preprocessor
        self.metadata = self._pipeline.metadata
        self.feature_columns = self._pipeline.feature_columns

    def _preprocess(self, customer: CustomerProfile) -> pd.DataFrame:
        """Transform customer profile into model-ready feature vector."""
        data = customer.model_dump()
        return self._pipeline.preprocess_customer(data)

    def _classify_risk(self, probability: float) -> RiskLevel:
        """Map probability to risk tier."""
        if probability < 0.3:
            return RiskLevel.LOW
        elif probability < 0.5:
            return RiskLevel.MEDIUM
        elif probability < 0.75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def _get_top_factors(self, features: pd.DataFrame, n: int = 5) -> list[str]:
        """Extract top risk factors weighted by feature values."""
        importances = dict(self.metadata.get("feature_importances", []))
        scores: dict[str, float] = {}
        for col in features.columns:
            if col in importances:
                val = abs(float(features[col].iloc[0]))
                scores[col] = importances[col] * (1 + val)
        sorted_factors = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        clean_names: list[str] = []
        for name, _ in sorted_factors[:n]:
            clean = name.replace("_", " ").replace("  ", " ")
            clean_names.append(clean)
        return clean_names

    def predict(self, customer: CustomerProfile) -> RiskPrediction:
        """
        Run churn prediction on a customer profile.

        This is the primary tool interface called by the ReAct loop.
        """
        try:
            features = self._preprocess(customer)
            probability = float(self.model.predict_proba(features)[:, 1][0])
            risk_level = self._classify_risk(probability)
            top_factors = self._get_top_factors(features)

            model_name = self.metadata.get("model_name", "Unknown")
            roc_auc = self.metadata.get("metrics", {}).get("roc_auc", 0)
            confidence = "High" if roc_auc >= 0.85 else "Medium" if roc_auc >= 0.75 else "Low"

            return RiskPrediction(
                churn_probability=round(probability, 4),
                risk_level=risk_level,
                top_risk_factors=top_factors,
                model_name=model_name,
                model_confidence=confidence,
            )
        except Exception as e:
            logger.error("Risk prediction failed: %s", e)
            raise


class ContextRetriever:
    """
    Simulates a CRM/database query to fetch customer context.

    In production, this would connect to a real database or CRM API.
    For the hackathon, it generates deterministic but realistic context
    based on the customer's profile features.
    """

    TICKET_TEMPLATES: list[dict[str, Any]] = [
        {
            "category": "Billing",
            "severity": "Medium",
            "summary": "Customer disputed a charge on their latest invoice",
        },
        {
            "category": "Technical",
            "severity": "High",
            "summary": "Frequent internet connectivity drops reported in the last 2 weeks",
        },
        {
            "category": "Service",
            "severity": "Low",
            "summary": "Inquiry about upgrading to a different internet plan",
        },
        {
            "category": "Technical",
            "severity": "Critical",
            "summary": "Complete service outage lasting more than 24 hours",
        },
        {
            "category": "Billing",
            "severity": "High",
            "summary": "Auto-payment failed twice, causing service suspension notice",
        },
        {
            "category": "Retention",
            "severity": "Medium",
            "summary": "Customer called to inquire about cancellation process",
        },
        {
            "category": "Technical",
            "severity": "Medium",
            "summary": "Slow internet speeds below contracted bandwidth",
        },
        {
            "category": "Service",
            "severity": "Low",
            "summary": "Request for detailed usage breakdown of streaming services",
        },
    ]

    def retrieve(self, customer: CustomerProfile, risk: RiskPrediction) -> CustomerContext:
        """
        Fetch customer context based on profile and risk assessment.

        Generates deterministic context using the customer's attributes
        to simulate what a real CRM lookup would return.
        """
        try:
            from datetime import timedelta
            rng = np.random.RandomState(hash(customer.customer_id) % (2**31))

            # ── Loyalty tier based on tenure ────────────────────────────────
            if customer.tenure >= 48:
                loyalty_tier = "Platinum"
            elif customer.tenure >= 24:
                loyalty_tier = "Gold"
            elif customer.tenure >= 12:
                loyalty_tier = "Silver"
            else:
                loyalty_tier = "Bronze"

            # ── Lifetime value ──────────────────────────────────────────────
            lifetime_value = round(customer.TotalCharges * 1.15, 2)

            # ── Generate support tickets (more for high-risk customers) ─────
            if risk.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                n_tickets = rng.randint(2, 5)
            elif risk.risk_level == RiskLevel.MEDIUM:
                n_tickets = rng.randint(1, 3)
            else:
                n_tickets = rng.randint(0, 2)

            tickets: list[SupportTicket] = []
            for i in range(n_tickets):
                template = self.TICKET_TEMPLATES[rng.randint(0, len(self.TICKET_TEMPLATES))]
                days_ago = rng.randint(1, 90)
                ticket_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
                tickets.append(SupportTicket(
                    ticket_id=f"TK-{rng.randint(10000, 99999)}",
                    date=ticket_date,
                    category=template["category"],
                    severity=template["severity"],
                    resolved=bool(rng.choice([True, False], p=[0.6, 0.4])),
                    summary=template["summary"],
                ))

            tickets.sort(key=lambda t: t.date, reverse=True)

            # ── Contract renewal ────────────────────────────────────────────
            if customer.Contract == "Month-to-month":
                renewal_days = rng.randint(1, 30)
            elif customer.Contract == "One year":
                renewal_days = rng.randint(30, 365)
            else:
                renewal_days = rng.randint(180, 730)
            renewal_date = (datetime.now() + timedelta(days=renewal_days)).strftime("%Y-%m-%d")

            # ── Payment issues ──────────────────────────────────────────────
            payment_issues = 0
            if customer.PaymentMethod == "Electronic check":
                payment_issues = rng.randint(1, 4)
            elif customer.PaymentMethod == "Mailed check":
                payment_issues = rng.randint(0, 2)

            # ── NPS Score ───────────────────────────────────────────────────
            if risk.risk_level == RiskLevel.CRITICAL:
                nps = rng.randint(0, 4)
            elif risk.risk_level == RiskLevel.HIGH:
                nps = rng.randint(3, 6)
            elif risk.risk_level == RiskLevel.MEDIUM:
                nps = rng.randint(5, 8)
            else:
                nps = rng.randint(7, 11)

            # ── Contextual notes ────────────────────────────────────────────
            notes: list[str] = []
            if customer.tenure < 6:
                notes.append("New customer -- still in onboarding phase")
            if customer.SeniorCitizen == 1:
                notes.append("Senior citizen -- may prefer phone support over digital")
            if customer.InternetService == "Fiber optic" and any(
                t.category == "Technical" for t in tickets
            ):
                notes.append("Fiber optic user with recent technical issues -- prioritize network team")
            if customer.Contract == "Month-to-month":
                notes.append("Month-to-month contract -- high flexibility, low switching cost")
            if payment_issues > 1:
                notes.append(f"Payment issues detected ({payment_issues} in last 90 days)")
            if customer.Dependents == "Yes":
                notes.append("Family plan potential -- may respond to family bundle offers")

            account_status = "Active"
            if payment_issues >= 3:
                account_status = "At Risk - Payment Issues"
            elif any(not t.resolved for t in tickets):
                account_status = "Active - Open Tickets"

            return CustomerContext(
                customer_id=customer.customer_id,
                account_status=account_status,
                loyalty_tier=loyalty_tier,
                lifetime_value=lifetime_value,
                last_interaction_days=rng.randint(1, 60),
                recent_tickets=tickets,
                contract_renewal_date=renewal_date,
                payment_issues=payment_issues,
                nps_score=nps,
                notes=notes,
            )
        except Exception as e:
            logger.error("Context retrieval failed: %s", e)
            raise


class RetentionProposalGenerator:
    """
    Evaluates company incentives against corporate policy constraints
    and produces a validated retention proposal.
    """

    def __init__(self) -> None:
        self.policy = CorporatePolicy()

    def _validate_discount(self, discount_pct: float) -> tuple[float, str | None]:
        """Clamp a proposed discount to the corporate policy maximum."""
        if discount_pct > self.policy.max_discount_pct:
            note = (
                f"Discount capped: requested {discount_pct:.0f}% exceeds "
                f"policy max of {self.policy.max_discount_pct:.0f}%"
            )
            return self.policy.max_discount_pct, note
        return discount_pct, None

    def _validate_free_months(self, months: int) -> tuple[int, str | None]:
        """Clamp proposed free months to the corporate policy maximum."""
        if months > self.policy.max_free_months:
            note = (
                f"Free months capped: requested {months} exceeds "
                f"policy max of {self.policy.max_free_months}"
            )
            return self.policy.max_free_months, note
        return months, None

    def generate(
        self,
        customer: CustomerProfile,
        risk: RiskPrediction,
        context: CustomerContext | None,
    ) -> RetentionProposal:
        """Generate and validate a retention proposal against corporate policies."""
        try:
            policy_notes: list[str] = []
            policy_compliant: bool = True
            actions: list[str] = []
            strategy: dict[str, Any] = {}

            if risk.risk_level == RiskLevel.CRITICAL:
                strategy["urgency"] = "Immediate"
                strategy["estimated_retention_lift"] = "30-45%"

                if customer.Contract == "Month-to-month":
                    raw_discount = 25 if (
                        context and context.loyalty_tier in ("Gold", "Platinum")
                    ) else 20
                    discount, note = self._validate_discount(raw_discount)
                    if note:
                        policy_notes.append(note)
                        policy_compliant = False

                    strategy["strategy_title"] = "Critical Retention: Premium Loyalty Save"
                    new_rate = customer.MonthlyCharges * (1 - discount / 100)
                    strategy["personalized_offer"] = (
                        f"Offer an immediate {discount:.0f}% discount on monthly charges "
                        f"(reducing from ${customer.MonthlyCharges:.2f} to "
                        f"${new_rate:.2f}/month) "
                        f"when the customer commits to a {self.policy.min_contract_months}-month contract. "
                    )
                    if context and any(
                        t.category == "Technical" for t in context.recent_tickets
                    ):
                        strategy["personalized_offer"] += (
                            "Additionally, provide a dedicated technical support line and "
                            "schedule a network quality assessment within 48 hours to address "
                            "their recent technical issues."
                        )
                    else:
                        free_months, note = self._validate_free_months(3)
                        if note:
                            policy_notes.append(note)
                        strategy["personalized_offer"] += (
                            f"Include {free_months} months of free premium streaming bundle "
                            f"as an added incentive."
                        )
                    actions.extend([
                        "Escalate to Senior Retention Specialist immediately",
                        f"Apply {discount:.0f}% loyalty discount upon contract commitment",
                        "Schedule personal follow-up call within 24 hours",
                        "Flag account for executive review if customer declines first offer",
                    ])
                else:
                    strategy["strategy_title"] = "Critical Retention: Contract Renewal Incentive"
                    free_months, note = self._validate_free_months(1)
                    if note:
                        policy_notes.append(note)
                    strategy["personalized_offer"] = (
                        f"Provide a complimentary service upgrade (free speed tier increase) "
                        f"and waive the next {free_months} month's charges "
                        f"(${customer.MonthlyCharges:.2f}) as a goodwill gesture."
                    )
                    actions.extend([
                        "Assign dedicated account manager",
                        f"Offer {free_months} free month(s) + service upgrade",
                        "Conduct satisfaction survey",
                    ])

            elif risk.risk_level == RiskLevel.HIGH:
                strategy["urgency"] = "Within 48 hours"
                strategy["estimated_retention_lift"] = "20-30%"

                has_tech_issues = context and any(
                    t.category == "Technical" and not t.resolved
                    for t in context.recent_tickets
                )
                has_billing_issues = context and context.payment_issues > 1

                if has_tech_issues:
                    raw_discount = 15
                    discount, note = self._validate_discount(raw_discount)
                    if note:
                        policy_notes.append(note)
                        policy_compliant = False

                    strategy["strategy_title"] = (
                        "Proactive Technical Resolution + Loyalty Reward"
                    )
                    savings = customer.MonthlyCharges * (discount / 100) * 6
                    strategy["personalized_offer"] = (
                        f"Apologize for the recent technical issues and offer a "
                        f"{discount:.0f}% discount for the next 6 months "
                        f"(saving ${savings:.2f} total). "
                        f"Dispatch a priority technician visit within 48 hours and "
                        f"provide a direct support line for future issues."
                    )
                    actions.extend([
                        "Dispatch priority technician within 48 hours",
                        f"Apply {discount:.0f}% discount for 6 months",
                        "Set up proactive network monitoring for this customer",
                        "Send personalized apology email from regional manager",
                    ])
                elif has_billing_issues:
                    raw_discount = 10
                    discount, note = self._validate_discount(raw_discount)
                    if note:
                        policy_notes.append(note)
                        policy_compliant = False

                    strategy["strategy_title"] = "Payment Flexibility Program"
                    new_rate = customer.MonthlyCharges * (1 - discount / 100)

                    waiver_note = None
                    if context and context.payment_issues > 0:
                        est_fees = context.payment_issues * 25.0
                        if est_fees > self.policy.max_fee_waiver_amount:
                            waiver_note = (
                                f"Fee waiver capped at ${self.policy.max_fee_waiver_amount:.2f} "
                                f"(estimated fees: ${est_fees:.2f})"
                            )
                            policy_notes.append(waiver_note)

                    strategy["personalized_offer"] = (
                        f"Offer flexible payment options including autopay with a "
                        f"{discount:.0f}% discount "
                        f"(new rate: ${new_rate:.2f}/month), "
                        f"and waive any late payment fees accumulated in the last 90 days."
                    )
                    actions.extend([
                        "Waive accumulated late fees",
                        f"Set up autopay with {discount:.0f}% incentive discount",
                        "Offer payment date flexibility",
                        "Follow up in 2 weeks to confirm payment stability",
                    ])
                else:
                    tenure_discount_raw = 15 if customer.tenure > 24 else 10
                    discount, note = self._validate_discount(tenure_discount_raw)
                    if note:
                        policy_notes.append(note)
                        policy_compliant = False

                    strategy["strategy_title"] = "Value Enhancement Bundle"
                    new_rate = customer.MonthlyCharges * (1 - discount / 100)
                    strategy["personalized_offer"] = (
                        f"Offer a {discount:.0f}% loyalty discount as a valued "
                        f"{'long-term ' if customer.tenure > 24 else ''}subscriber, "
                        f"bringing the monthly rate to ${new_rate:.2f}. "
                        f"Bundle with free device protection and tech support for 6 months."
                    )
                    actions.extend([
                        f"Apply {discount:.0f}% loyalty discount",
                        "Add complimentary device protection (6 months)",
                        "Add complimentary tech support (6 months)",
                        "Schedule satisfaction check-in for 30 days",
                    ])

            elif risk.risk_level == RiskLevel.MEDIUM:
                strategy["urgency"] = "This week"
                strategy["estimated_retention_lift"] = "10-20%"
                strategy["strategy_title"] = "Proactive Engagement & Value Communication"

                raw_discount = 5
                discount, note = self._validate_discount(raw_discount)
                if note:
                    policy_notes.append(note)
                    policy_compliant = False

                competitor_savings = max(0, customer.TotalCharges * 0.08)
                strategy["personalized_offer"] = (
                    f"Send a personalized value summary showing the customer has saved "
                    f"${competitor_savings:.2f} compared to competitors. "
                    f"Offer a {discount:.0f}% discount if they switch to autopay and a "
                    f"free service upgrade trial for 3 months."
                )
                actions.extend([
                    "Send personalized value communication email",
                    f"Offer {discount:.0f}% autopay discount",
                    "Provide free 3-month service upgrade trial",
                    "Add to proactive outreach queue",
                ])

            else:  # LOW risk
                strategy["urgency"] = "This month"
                strategy["estimated_retention_lift"] = "5-10%"
                strategy["strategy_title"] = "Loyalty Appreciation & Engagement"
                strategy["personalized_offer"] = (
                    f"Send a loyalty appreciation message recognizing "
                    f"{customer.tenure} months of partnership. "
                    f"Offer early access to new features and a referral bonus program "
                    f"(${self.policy.max_referral_credit:.0f} credit per successful referral)."
                )
                actions.extend([
                    "Send loyalty appreciation communication",
                    "Enroll in referral bonus program",
                    "Add to quarterly satisfaction survey list",
                ])

            if customer.Dependents == "Yes" and risk.risk_level in (
                RiskLevel.HIGH, RiskLevel.CRITICAL
            ):
                actions.append("Present family bundle options with multi-line discounts")

            if customer.SeniorCitizen == 1:
                actions.append("Offer dedicated senior support line with extended hours")

            if not policy_notes:
                policy_notes.append("All proposal parameters within corporate policy limits")
                policy_compliant = True

            return RetentionProposal(
                strategy_title=strategy.get("strategy_title", "Retention Strategy"),
                personalized_offer=strategy.get("personalized_offer", ""),
                recommended_actions=actions,
                estimated_retention_lift=strategy.get("estimated_retention_lift", "N/A"),
                urgency=strategy.get("urgency", "This week"),
                policy_compliant=policy_compliant,
                policy_notes=policy_notes,
            )

        except Exception as e:
            logger.error("Retention proposal generation failed: %s", e)
            raise


class ReActOrchestrator:
    """
    The core reasoning agent — backward-compatible wrapper.

    Maintains the same interface used by tests while internally delegating
    to the new three-layer pipeline architecture.

    CPU-optimized: All operations target Intel i5 / Iris Xe.
    """

    def __init__(self) -> None:
        self.risk_predictor = RiskPredictor()
        self.context_retriever = ContextRetriever()
        self.proposal_generator = RetentionProposalGenerator()
        self.llm_client: Any | None = None
        self.llm_available: bool = False
        self._init_llm()

    def _init_llm(self) -> None:
        """Initialize the LLM client if a GitHub token is available."""
        github_token = os.getenv("GITHUB_TOKEN")
        if github_token and github_token.strip() and github_token != "ghp_your_token_here":
            try:
                from openai import OpenAI
                self.llm_client = OpenAI(
                    base_url="https://models.inference.ai.azure.com",
                    api_key=github_token,
                )
                self.llm_available = True
                logger.info("LLM reasoning engine initialized (GitHub Models)")
            except ImportError:
                logger.warning(
                    "openai package not installed. Using rule-based reasoning."
                )
            except Exception as e:
                logger.warning(
                    "LLM initialization failed: %s. Using rule-based reasoning.", e
                )
        else:
            logger.info("No GITHUB_TOKEN found. Using rule-based reasoning engine.")

    def run(self, customer: CustomerProfile) -> RetentionStrategy:
        """
        Execute the full ReAct reasoning loop for a customer.

        Uses the rule-based deterministic pipeline to ensure test compatibility.
        """
        trace: list[ReasoningStep] = []
        step = 0

        # ── STEP 1: Thought ─────────────────────────────────────────────
        step += 1
        thought_1 = (
            f"I need to evaluate churn risk for customer {customer.customer_id}. "
            f"They are on a {customer.Contract} contract with {customer.tenure} months tenure, "
            f"paying ${customer.MonthlyCharges:.2f}/month via {customer.PaymentMethod}. "
            f"Internet service: {customer.InternetService}. "
            f"I will call the Risk Predictor tool to get a quantitative churn score."
        )
        trace.append(ReasoningStep(step_number=step, phase="Thought", content=thought_1))
        logger.info("[Thought %d] %s", step, thought_1)

        # ── STEP 2: Action — Risk Predictor ─────────────────────────────
        step += 1
        trace.append(ReasoningStep(
            step_number=step,
            phase="Action",
            content="Calling Risk Predictor tool with customer profile...",
        ))
        logger.info("[Action  %d] Invoking Risk Predictor tool", step)

        risk: RiskPrediction = self.risk_predictor.predict(customer)

        # ── STEP 3: Observation ─────────────────────────────────────────
        step += 1
        obs_1 = (
            f"Risk Predictor returned: churn_probability={risk.churn_probability:.1%}, "
            f"risk_level={risk.risk_level.value}, "
            f"top_factors=[{', '.join(risk.top_risk_factors[:3])}], "
            f"model={risk.model_name} (confidence: {risk.model_confidence})."
        )
        trace.append(ReasoningStep(step_number=step, phase="Observation", content=obs_1))
        logger.info("[Observe %d] %s", step, obs_1)

        # ── STEP 4: Thought — Context decision ─────────────────────────
        step += 1
        context: CustomerContext | None = None

        if risk.churn_probability > 0.30:
            thought_2 = (
                f"The churn probability is {risk.churn_probability:.1%} (>{30}% threshold). "
                f"This is a {risk.risk_level.value}-risk customer. "
                f"Calling Context Retriever to fetch support tickets, contract status, "
                f"and interaction history."
            )
            trace.append(ReasoningStep(step_number=step, phase="Thought", content=thought_2))
            logger.info("[Thought %d] %s", step, thought_2)

            step += 1
            trace.append(ReasoningStep(
                step_number=step,
                phase="Action",
                content="Calling Context Retriever tool to fetch customer history...",
            ))
            logger.info("[Action  %d] Invoking Context Retriever tool", step)

            context = self.context_retriever.retrieve(customer, risk)

            step += 1
            ticket_summary = (
                f"{len(context.recent_tickets)} recent tickets "
                f"({sum(1 for t in context.recent_tickets if not t.resolved)} unresolved)"
            ) if context.recent_tickets else "No recent tickets"

            obs_2 = (
                f"Context retrieved: loyalty_tier={context.loyalty_tier}, "
                f"lifetime_value=${context.lifetime_value:.2f}, "
                f"account_status='{context.account_status}', "
                f"payment_issues={context.payment_issues}, "
                f"NPS={context.nps_score}/10, "
                f"tickets={ticket_summary}."
            )
            trace.append(ReasoningStep(step_number=step, phase="Observation", content=obs_2))
            logger.info("[Observe %d] %s", step, obs_2)
        else:
            thought_2 = (
                f"The churn probability is {risk.churn_probability:.1%} (below 30% threshold). "
                f"This is a low-risk customer. Generating a light-touch engagement strategy."
            )
            trace.append(ReasoningStep(step_number=step, phase="Thought", content=thought_2))
            logger.info("[Thought %d] %s", step, thought_2)

        # ── STEP 7: Thought — Generate proposal ────────────────────────
        step += 1
        trace.append(ReasoningStep(
            step_number=step,
            phase="Thought",
            content=(
                f"Generating a personalized retention strategy "
                f"for this {risk.risk_level.value}-risk customer using "
                f"the Retention Proposal Generator tool."
            ),
        ))
        logger.info("[Thought %d] Preparing to generate retention proposal", step)

        # ── STEP 8: Action — Proposal Generator ────────────────────────
        step += 1
        trace.append(ReasoningStep(
            step_number=step,
            phase="Action",
            content="Calling Retention Proposal Generator tool to validate and compile strategy...",
        ))
        logger.info("[Action  %d] Invoking Retention Proposal Generator tool", step)

        proposal: RetentionProposal = self.proposal_generator.generate(
            customer, risk, context
        )

        # ── STEP 9: Observation — Proposal result ──────────────────────
        step += 1
        policy_status = "COMPLIANT" if proposal.policy_compliant else "ADJUSTED"
        trace.append(ReasoningStep(
            step_number=step,
            phase="Observation",
            content=(
                f"Retention proposal generated: '{proposal.strategy_title}' "
                f"(policy: {policy_status}). "
                f"Urgency: {proposal.urgency}. "
                f"Estimated retention lift: {proposal.estimated_retention_lift}. "
                f"Actions: {len(proposal.recommended_actions)}. "
                f"Policy notes: {'; '.join(proposal.policy_notes)}."
            ),
        ))
        logger.info("[Observe %d] Proposal: %s (%s)", step, proposal.strategy_title, policy_status)

        # ── Build reasoning summary ─────────────────────────────────────
        reasoning_parts: list[str] = [
            f"Customer {customer.customer_id} assessed at {risk.churn_probability:.1%} churn risk "
            f"({risk.risk_level.value}).",
            f"Key risk factors: {', '.join(risk.top_risk_factors[:3])}.",
        ]
        if context:
            reasoning_parts.append(
                f"Context: {context.loyalty_tier} tier, "
                f"LTV=${context.lifetime_value:.2f}, "
                f"{len(context.recent_tickets)} tickets, "
                f"NPS={context.nps_score}/10."
            )
        reasoning_parts.append(
            f"Strategy: {proposal.strategy_title} "
            f"(policy: {policy_status})."
        )

        return RetentionStrategy(
            customer_id=customer.customer_id,
            risk_level=risk.risk_level,
            churn_probability=risk.churn_probability,
            strategy_title=proposal.strategy_title,
            personalized_offer=proposal.personalized_offer,
            reasoning_summary=" ".join(reasoning_parts),
            recommended_actions=proposal.recommended_actions,
            estimated_retention_lift=proposal.estimated_retention_lift,
            urgency=proposal.urgency,
            reasoning_trace=trace,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    interactive_cli()
