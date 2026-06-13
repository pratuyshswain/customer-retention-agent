# Customer Churn Prediction & Retention Agent

> **Microsoft Agents League Hackathon** | GIET Baniatangi / BPUT  
> Built with Python, scikit-learn, FastAPI, and the ReAct Agent Framework

---

## Overview

An end-to-end Customer Churn Prediction system upgraded with an intelligent **Reasoning Agent** that autonomously:
1. **Predicts** churn probability using a trained ML model (Tool 1: Risk Predictor)
2. **Investigates** why a customer might churn by querying contextual data (Tool 2: Context Retriever)
3. **Generates** personalized retention strategies via a multi-step ReAct reasoning loop

## Architecture

```
+─────────────────────────────────────────────────────────+
|              ReAct Orchestrator (Agent Core)             |
|                                                         |
|  Thought → Action → Observation → Action → Observation  |
|                                                         |
|  ┌──────────────────┐    ┌─────────────────────────┐    |
|  │  Tool 1: Risk    │    │  Tool 2: Context        │    |
|  │  Predictor       │    │  Retriever              │    |
|  │  (ML Model)      │    │  (CRM Simulator)        │    |
|  └──────────────────┘    └─────────────────────────┘    |
|                                                         |
|  ┌──────────────────────────────────────────────────┐   |
|  │  LLM Reasoning (OpenAI) / Rule-Based Fallback   │   |
|  └──────────────────────────────────────────────────┘   |
+─────────────────────────────────────────────────────────+
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
      FastAPI REST   Retention    Evaluation
      Endpoint       Strategy     Report
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the ML Pipeline

```bash
python data/generate_dataset.py      # Generate synthetic Telco dataset
python -X utf8 src/data_engineer.py   # EDA + preprocessing
python -X utf8 src/ml_modeler.py      # Train & evaluate models
```

### 3. Run the Reasoning Agent

```bash
python -X utf8 src/agent_orchestrator.py
```

### 4. Start the API Server

```bash
python run_server.py
# Swagger UI: http://localhost:8000/docs
```

### 5. (Optional) Enable LLM Reasoning

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
# The agent works without it (rule-based fallback)
```

## Model Performance

| Metric | Target | Achieved |
|--------|:------:|:--------:|
| ROC-AUC | >= 0.82 | **0.8883** |
| F1 Score | >= 0.60 | **0.7047** |
| Recall | >= 0.70 | **0.8284** |

## Agent Demo Output

```
  Customer     Risk       Churn%   Urgency            Strategy
  ------------ ---------- -------- ------------------
  DEMO-001     Critical   98.4%    Immediate          Critical Retention: Premium Loyalty Save
  DEMO-002     Low        0.2%     This month         Loyalty Appreciation & Engagement
  DEMO-003     Critical   96.2%    Immediate          Critical Retention: Premium Loyalty Save
```

## Test Suite

```bash
python -m pytest tests/ -v
# 57 tests, 0 failures
```

## Project Structure

```
├── data/
│   ├── generate_dataset.py     # Synthetic data generator
│   ├── telco_churn.csv         # Raw dataset (7043×21)
│   └── processed_churn.csv     # Preprocessed features
├── src/
│   ├── agent_orchestrator.py   # ReAct reasoning agent
│   ├── data_engineer.py        # EDA + preprocessing
│   ├── ml_modeler.py           # Model training
│   └── api.py                  # FastAPI endpoints
├── models/                     # Serialized model artifacts
├── tests/                      # pytest test suite
├── artifacts/                  # Reports & visualizations
├── .env.example                # Environment template
├── requirements.txt
├── run_server.py
└── README.md
```

## Tech Stack

- **ML**: scikit-learn (Logistic Regression, Random Forest + GridSearchCV)
- **Agent Framework**: Custom ReAct implementation with Pydantic schemas
- **LLM**: OpenAI GPT-4o-mini (optional, with rule-based fallback)
- **API**: FastAPI + Uvicorn
- **Data**: pandas, numpy, matplotlib, seaborn

## License

MIT
