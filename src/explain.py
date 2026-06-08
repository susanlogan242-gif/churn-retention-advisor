"""Per-customer SHAP explanations (M3).

Expose top_drivers(model, X, row) -> list of (feature, direction, magnitude):
the features most pushing a given customer's churn risk up. These drivers are
the *grounding* fed to the LLM advisor.
"""

# TODO M3: shap.TreeExplainer over the fitted model; helper returning ranked drivers

if __name__ == "__main__":
    raise SystemExit("Not implemented yet — see notebooks/churn_advisor.ipynb")
