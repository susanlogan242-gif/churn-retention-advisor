"""Streamlit app (M5).

- Sidebar: model metrics + dataset summary.
- Ranked at-risk list sorted by £-at-risk (prob x value), filterable.
- Customer detail: risk gauge, SHAP driver chart, and the LLM retention play.
- A generic-vs-grounded toggle (the demo money-shot).

Run: streamlit run src/app.py
"""

# TODO M5: build the app once train.py / explain.py / llm_advisor.py exist.

import streamlit as st

st.set_page_config(page_title="Churn + Retention Advisor", page_icon="📉", layout="wide")
st.title("Churn Predictor + Retention Advisor")
st.info("🚧 Coming together in M5 — prototype in notebooks/churn_advisor.ipynb first.")
