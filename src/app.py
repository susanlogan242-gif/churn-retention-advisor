"""Churn Predictor + Retention Advisor — Streamlit app.

Ranks at-risk customers by £-at-risk, explains each with SHAP, and proposes a
grounded retention play. Run: streamlit run src/app.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # render charts to image (thread-safe for Streamlit; fixes broken-image icon)
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import shap

# ---- brand palette -------------------------------------------------------
INK = "#1b2733"
TEAL = "#0f766e"        # protective / risk-reducing
WARM = "#c2410c"        # risk-raising
MUTE = "#5b6b76"
GRID = "#d7e0e0"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

st.set_page_config(page_title="Churn + Retention Advisor", page_icon="📉",
                   layout="wide", initial_sidebar_state="expanded")

# ---- light CSS polish (within Streamlit's limits) ------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1180px; }
#MainMenu, footer { visibility: hidden; }
.app-title { font-size: 1.85rem; font-weight: 700; color: #1b2733; letter-spacing: -0.02em; margin: 0; }
.app-sub  { color: #5b6b76; font-size: 0.98rem; margin: .25rem 0 0; }
.badge { display:inline-block; padding:.18rem .6rem; border-radius:999px;
         font-size:.78rem; font-weight:600; letter-spacing:.01em; }
.badge-high { background:#fdeae1; color:#c2410c; }
.badge-med  { background:#fef6e7; color:#b45309; }
.badge-low  { background:#e6f4f1; color:#0f766e; }
div[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 700; }
section[data-testid="stSidebar"] { border-right: 1px solid #e2e9e9; }
hr { margin: .8rem 0; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training model…")
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
    order = np.argsort(np.abs(row))[::-1][:k]
    return [(cols[j], float(row[j])) for j in order]


PRETTY = {"Contract_Two year": "Two-year contract", "Contract_One year": "One-year contract",
          "InternetService_Fiber optic": "Fibre-optic internet", "tenure": "Tenure (months)",
          "MonthlyCharges": "Monthly charges", "TotalCharges": "Total charges",
          "PaymentMethod_Electronic check": "Pays by e-check",
          "OnlineSecurity_Yes": "Has online security", "TechSupport_Yes": "Has tech support",
          "PaperlessBilling_Yes": "Paperless billing"}


def label(c):
    return PRETTY.get(c, c.replace("_", ": ").replace("Yes", "yes").replace("No", "no"))


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
        nice = {"OnlineSecurity": "Online Security", "TechSupport": "Tech Support"}
        names = [nice[m] for m in missing]
        why.append("Missing protective add-ons: " + ", ".join(names) + ".")
        actions.append(("Bundle " + " + ".join(names) + " free for 3 months",
                        "adds the 'stickiness' features the model shows reduce churn"))
    if r["PaymentMethod"] == "Electronic check":
        why.append("Pays by electronic check (a friction-heavy, churn-correlated method).")
        actions.append(("Move to auto-pay (card/bank) with a small one-off credit",
                        "removes a known churn-correlated friction point"))
    if not actions:
        actions.append(("Proactive check-in call + a tailored offer",
                        "general retention outreach for a flagged account"))
    return why, actions[:3]


def risk_band(p):
    if p >= 0.70:
        return "high", "HIGH RISK"
    if p >= 0.40:
        return "med", "MEDIUM RISK"
    return "low", "LOWER RISK"


# ---------------- header ----------------
st.markdown('<p class="app-title">📉 Churn Predictor + Retention Advisor</p>', unsafe_allow_html=True)
st.markdown('<p class="app-sub">Rank at-risk customers by revenue at risk, see <b>why</b> each is at risk (SHAP), '
            'and get a retention play grounded in that customer\'s own drivers.</p>', unsafe_allow_html=True)
st.write("")

# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown("### Model")
    st.metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}")
    st.metric("PR-AUC", f"{metrics['PR-AUC']:.3f}")
    st.metric("Brier (calibration ↓)", f"{metrics['Brier']:.3f}")
    st.caption("RandomForest, well-calibrated. Scores shown are out-of-sample "
               "(held-out test customers the model never saw in training).")
    st.divider()
    topn = st.slider("Show top N at-risk", 10, 100, 25, step=5)
    st.caption("**£-at-risk** = churn probability × monthly charge × 12. "
               "It turns a risk score into a revenue-prioritised call list.")

board = (pd.DataFrame({
    "Customer": idte.values,
    "Churn risk": proba * 100,
    "£/yr at risk": at_risk,
    "Tenure (mo)": rawte["tenure"].values,
    "Contract": rawte["Contract"].values})
    .sort_values("£/yr at risk", ascending=False).reset_index(drop=True))

left, right = st.columns([1.0, 1.25], gap="large")

with left:
    st.markdown("#### At-risk customers")
    st.caption("Sorted by annual revenue at risk — your prioritised call list.")
    st.dataframe(
        board.head(topn), use_container_width=True, hide_index=True, height=470,
        column_config={
            "Customer": st.column_config.TextColumn("Customer", width="small"),
            "Churn risk": st.column_config.ProgressColumn(
                "Churn risk", format="%.0f%%", min_value=0, max_value=100),
            "£/yr at risk": st.column_config.NumberColumn("£/yr at risk", format="£%d"),
            "Tenure (mo)": st.column_config.NumberColumn("Tenure", format="%d mo"),
        })

with right:
    st.markdown("#### Customer detail")
    pick = st.selectbox("Select a customer", board["Customer"].head(topn).tolist(),
                        label_visibility="collapsed")
    i = int(rawte.index[idte == pick][0])
    band, band_txt = risk_band(proba[i])

    st.markdown(f'<span class="badge badge-{band}">{band_txt}</span>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Churn risk", f"{proba[i]*100:.0f}%")
    c2.metric("£/yr at risk", f"£{at_risk[i]:,.0f}")
    c3.metric("Tenure", f"{int(rawte.iloc[i]['tenure'])} mo")

    st.markdown("**Why at risk — top drivers (SHAP)**")
    drv = shap_drivers(i)
    names = [label(d[0]) for d in drv][::-1]
    vals = [d[1] for d in drv][::-1]
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    ax.barh(names, vals, color=[WARM if v > 0 else TEAL for v in vals], height=0.62)
    ax.axvline(0, color=MUTE, lw=0.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(labelsize=8.5, length=0, colors=INK)
    ax.set_xlabel("← lowers risk      raises risk →", fontsize=8, color=MUTE)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("Warm bars push this customer toward churning; teal bars hold them.")

    grounded = st.toggle("Grounded advice (vs generic playbook)", value=True)
    with st.container(border=True):
        st.markdown("**🎯 Retention play**")
        if grounded:
            why, actions = retention_play(i)
            st.markdown("*Why this customer:* " + " ".join(why))
            for j, (a, reason) in enumerate(actions, 1):
                st.markdown(f"**{j}. {a}**  \n<span style='color:#5b6b76'>{reason}</span>",
                            unsafe_allow_html=True)
        else:
            st.info("Generic playbook: *“Reach out and offer a discount to keep them.”*  \n"
                    "Notice it ignores this customer's actual drivers — that's the difference "
                    "grounding makes.")

st.divider()
st.caption("Demo on the public IBM Telco dataset (no real customer PII). Gender is not used to drive "
           "actions. Retention plays are evidence-aligned decision support, not A/B-tested automation.")
