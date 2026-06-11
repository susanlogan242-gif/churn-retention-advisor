# Results — Churn Predictor + Retention Advisor

Worked results from the notebook (`notebooks/churn_advisor.ipynb`), run on the Telco Customer Churn dataset (7,043 customers; ~26.5% churn).

## Model comparison (ranked by PR-AUC + calibration)

| Model | ROC-AUC | PR-AUC | Brier ↓ | Recall@10% |
|---|--:|--:|--:|--:|
| **RandomForest** ✅ | **0.843** | **0.655** | **0.136** | 0.289 |
| LogReg | 0.843 | 0.636 | 0.138 | 0.283 |
| LightGBM | 0.831 | 0.624 | 0.143 | 0.278 |
| HistGradientBoosting | 0.829 | 0.628 | 0.144 | 0.273 |

**Chosen model: RandomForest** — best PR-AUC, best-calibrated (lowest Brier), competitive top-decile recall. ROC-AUC ≈ 0.84 is in line with the known ceiling for this dataset (chasing higher usually signals leakage).

**Calibration:** with `class_weight='balanced'` the original model was over-confident (over-estimated risk at the high end). Training without it left RandomForest **already well-calibrated** (raw Brier 0.136 = calibrated 0.136; curve hugs the diagonal), so no separate calibration layer is needed. Probabilities are trustworthy for the £-at-risk ranking.

**Top-decile precision ≈ 77%** (0.289 × 374 ÷ 141) — of the highest-risk customers flagged, ~3 in 4 are genuine churners: a strong "who to call first" list.

## What drives churn (global SHAP)
In order of impact: **contract type** (two-year / one-year strongly *reduce* risk), **tenure** (longer = lower risk), **MonthlyCharges** (higher = higher risk), **Fiber-optic internet** (higher risk), **Electronic-check payment** (higher risk), and **OnlineSecurity / TechSupport = Yes** (reduce risk). These are the canonical, sensible Telco drivers — the model is learning real signal.

## The at-risk segment (key insight)
Every top-risk customer shares the **same signature: fibre-optic + month-to-month + electronic check + missing security/support add-ons.** This isn't a set of individual problems — it's **one high-value churn segment** with **one playbook**: annual-contract offer + free security/support bundle + auto-pay nudge.

## Worked retention plays (LLM layer, grounded in each customer's SHAP drivers)

### Customer 2631 — 98% risk · ~£1,170/yr · HIGH
New, high-paying fibre customer with no commitment and no sticky add-ons.
1. 12-month contract + 15–20% loyalty discount (fixes month-to-month; eases the £99 bill).
2. Free TechSupport + OnlineSecurity for 3 months (adds the protective add-ons the model flags).
3. Move to auto-pay with a small credit (removes electronic-check friction).

### Customer 1 — 96% risk · ~£1,224/yr · HIGH
Brand-new (3 months), month-to-month, high £105.90/mo fibre, no TechSupport/OnlineSecurity.
1. 12-month contract + loyalty discount (fixes no-commitment in the fragile first quarter).
2. Free TechSupport + OnlineSecurity for 3 months.
3. Auto-pay switch + small credit.

### Customer 2 — 91% risk · ~£1,215/yr · HIGH
~11 months, high overall spend, month-to-month, high-cost fibre. **Already has TechSupport** → gap is OnlineSecurity.
1. Annual contract at a loyalty rate (reward the spend, lock it in).
2. Add OnlineSecurity free for 3 months (don't re-offer TechSupport — already held).
3. Auto-pay switch + small credit.

### Customer 3 — 93% risk · ~£1,172/yr · HIGH
6 months, month-to-month, high-cost fibre, no TechSupport/OnlineSecurity, paperless.
1. 12-month contract + loyalty discount.
2. Bundle TechSupport + OnlineSecurity free for 3 months.
3. Auto-pay switch + one-off credit.

*Each action references a real SHAP driver for that customer — no invented data. In the deployed app this is generated automatically (optionally hardened with a no-hallucination guardrail + the Anthropic API).*
