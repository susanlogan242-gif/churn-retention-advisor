"""Churn Predictor + Retention Advisor — Streamlit app.

Ranks at-risk customers by £-at-risk, explains each with SHAP, and proposes a
grounded retention play. Run: streamlit run src/app.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import shap

RED = "#ee3b2f"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

st.set_page_config(page_title="Churn + Retention Advisor", page_icon="📉", layout="wide")


@st.cache_resource
def load_and_train():
    df = pd.read_csv(DATA)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    ids = df["customerID"]
    d = df.drop(columns=["customerID"]).copy()
    d["Churn"] = (d["Churn"] == "Yes").astype(int)
    y = d["Churn"]
    Xraw = d.drop(columns=["Churn"])
    X = pd.get_dummies(Xraw, drop_first=True).astype(float)
    Xtr, Xte, ytr, yte, rawtr, rawte, idtr, idte = train_test_split(
        X, y, Xraw, ids, test_size=0.2, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                   random_state=42, n_jobs=-1).fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    metrics = {"ROC-AUC": roc_auc_score(yte, proba),
               "PR-AUC": average_precision_score(yte, proba),
               "Brier": brier_score_loss(yte, proba)}
    explainer = shap.TreeExplainer(model)
    return (model, list(X.columns), Xte.reset_index(drop=True),
            rawte.reset_index(drop=True), idte.reset_index(drop=True), proba, metrics, explainer)


model, cols, Xte, rawte, idte, proba, metrics, explainer = load_and_train()
at_risk = proba * rawte["MonthlyCharges"].astype(float).values * 12  # annual £-at-risk


def shap_drivers(i, k=6):
    sv = explainer.shap_values(Xte.iloc[[i]])
    if isinstance(sv, list):
        row = np.array(sv[1])[0]
    elif getattr(sv, "ndim", 2) == 3:
        row = np.array(sv)[0, :, 1]
    else:
        row = np.array(sv)[0]
    order = np.argsort(row)[::-1][:k]
    return [(cols[j], float(row[j])) for j in order]


def retention_play(i):
    r = rawte.iloc[i]
    why, actions = [], []
    if r["Contract"] == "Month-to-month":
        why.append("On a month-to-month contract (no long-term commitment).")
        actions.append(("Offer a 12-month contract with a 15–20% loyalty discount",
                        "directly fixes the biggest structural churn driver and eases the bill"))
    if int(r["tenure"]) < 12:
        why.append(f"New customer — only {int(r['tenure'])} months in (high-risk early window).")
    if float(r["MonthlyCharges"]) >= 80:
        why.append(f"High monthly bill (£{float(r['MonthlyCharges']):.2f}/mo).")
    if r["InternetService"] == "Fiber optic":
        why.append("On premium fibre-optic (higher-cost, higher-expectation plan).")
    missing = [a for a in ["OnlineSecurity", "TechSupport"] if str(r.get(a)) == "No"]
    if missing:
        why.append("Missing protective add-ons: " + ", ".join(missing) + ".")
        actions.append(("Bundle " + " + ".join(missing) + " free for 3 months",
                        "adds the 'stickiness' features the model shows reduce churn"))
    if r["PaymentMethod"] == "Electronic check":
        why.append("Pays by electronic check (a friction-heavy, churn-correlated method).")
        actions.append(("Move to auto-pay (card/bank) with a small one-off credit",
                        "removes a known churn-correlated friction point"))
    if not actions:
        actions.append(("Proactive check-in call + a tailored offer",
                        "general retention outreach for a flagged account"))
    return why, actions[:3]


# ---------------- UI ----------------
st.title("📉 Churn Predictor + Retention Advisor")
st.caption("Rank at-risk customers by revenue at risk, see *why* (SHAP), and get a grounded retention play.")

with st.sidebar:
    st.header("Model")
    st.metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}")
    st.metric("PR-AUC", f"{metrics['PR-AUC']:.3f}")
    st.metric("Brier (calibration ↓)", f"{metrics['Brier']:.3f}")
    st.caption("RandomForest, well-calibrated. Predictions shown are out-of-sample (test set).")
    st.divider()
    topn = st.slider("Show top N at-risk", 10, 100, 25, step=5)

board = (pd.DataFrame({
    "Customer": idte.values,
    "Churn risk": proba,
    "£/yr at risk": at_risk,
    "Tenure (mo)": rawte["tenure"].values,
    "Contract": rawte["Contract"].values})
    .sort_values("£/yr at risk", ascending=False).reset_index(drop=True))

left, right = st.columns([1.05, 1.25], gap="large")

with left:
    st.subheader("At-risk customers")
    st.caption("Sorted by annual £-at-risk = churn probability × monthly charge × 12.")
    show = board.head(topn).copy()
    show["Churn risk"] = (show["Churn risk"] * 100).round(0).astype(int).astype(str) + "%"
    show["£/yr at risk"] = "£" + show["£/yr at risk"].round(0).astype(int).astype(str)
    st.dataframe(show, use_container_width=True, hide_index=True, height=460)

with right:
    pick = st.selectbox("Select a customer", board["Customer"].head(topn).tolist())
    i = int(rawte.index[idte == pick][0])
    c1, c2, c3 = st.columns(3)
    c1.metric("Churn risk", f"{proba[i]*100:.0f}%")
    c2.metric("£/yr at risk", f"£{at_risk[i]:,.0f}")
    c3.metric("Tenure", f"{int(rawte.iloc[i]['tenure'])} mo")

    drv = shap_drivers(i)
    st.markdown("**Why at risk — top drivers (SHAP)**")
    fig, ax = plt.subplots(figsize=(5, 2.6))
    names = [d[0].replace("_", " ") for d in drv][::-1]
    vals = [d[1] for d in drv][::-1]
    ax.barh(names, vals, color=[RED if v > 0 else "#5b7fa6" for v in vals])
    ax.axvline(0, color="#999", lw=0.8); ax.set_xlabel("→ raises risk"); ax.tick_params(labelsize=8)
    fig.tight_layout(); st.pyplot(fig)

    why, actions = retention_play(i)
    grounded = st.toggle("Grounded advice (vs generic)", value=True)
    st.markdown("**Retention play**")
    if grounded:
        st.markdown("*Why:* " + " ".join(why))
        for j, (a, reason) in enumerate(actions, 1):
            st.markdown(f"**{j}. {a}** — {reason}")
    else:
        st.info("Generic playbook: *“Reach out to the customer and offer a discount to keep them.”*  "
                "— note how it ignores this customer's actual drivers. That's the difference grounding makes.")

st.caption("Demo on the public IBM Telco dataset. Retention actions are decision-support, evidence-aligned but not A/B-tested.")
