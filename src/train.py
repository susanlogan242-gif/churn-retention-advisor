"""Train + evaluate the churn model, then persist it.

M1-M2: clean the Telco data, train a baseline (logistic regression) then a
gradient-boosted model, evaluate honestly (ROC-AUC, PR-AUC, recall@top-decile,
calibration), and save the fitted model to models/churn_model.joblib.

Prototype interactively in notebooks/churn_advisor.ipynb first, then move the
settled logic here.
"""

# TODO M1: load data/WA_Fn-UseC_-Telco-Customer-Churn.csv, clean TotalCharges, encode
# TODO M2: train + evaluate + calibrate, joblib.dump the model

if __name__ == "__main__":
    raise SystemExit("Not implemented yet — see notebooks/churn_advisor.ipynb")
