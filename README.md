# Churn Predictor + Retention Advisor

A deployed tool that predicts which customers are about to churn **and** — for each at-risk customer — uses an LLM to turn the model's own reasoning into a specific, grounded retention play a Customer Success team could act on today.

> Portfolio project demonstrating a **classical-ML + LLM hybrid**: a calibrated churn model, per-customer SHAP explanations, and an LLM layer that converts those explanations into prioritised, money-aware retention actions — shipped as a Streamlit app.

## What makes it more than a churn notebook
1. **Grounded LLM advice.** The retention recommendations are built from each customer's actual SHAP drivers (not generic tips), with a guardrail that rejects advice not tied to a real driver.
2. **It ships.** A deployed Streamlit app, not a notebook.
3. **It speaks money.** Every prediction is tied to **£-at-risk** (churn probability × customer value) for a prioritised action list.

## Stack
Python · pandas · scikit-learn / LightGBM · SHAP · Claude API (`anthropic`, with prompt caching) · Streamlit · Streamlit Community Cloud

## Data
[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample, ~7,043 rows). Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in `data/` (not committed).

## Project layout
```
src/
  train.py         # clean data, train + evaluate the churn model, persist it
  explain.py       # per-customer SHAP drivers
  llm_advisor.py   # grounded LLM retention plays (structured JSON output)
  app.py           # Streamlit app
data/              # dataset (gitignored)
notebooks/         # EDA
```

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
python src/train.py
streamlit run src/app.py
```

## Status
🚧 In progress — see the build plan (M0 → M6).

## Build milestones
- [ ] **M0** Setup
- [ ] **M1** EDA + baseline model
- [ ] **M2** Strong model + honest evaluation (ROC-AUC, PR-AUC, calibration)
- [ ] **M3** Per-customer SHAP explainability
- [ ] **M4** Grounded LLM retention layer + hallucination guardrail
- [ ] **M5** Streamlit app (ranked £-at-risk list, customer detail, generic-vs-grounded toggle)
- [ ] **M6** Deploy + write-up
