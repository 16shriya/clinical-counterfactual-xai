import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Trustworthy Diabetes XAI Framework", layout="wide", page_icon="🩺str"
)

# ============================================================
# LOAD MODEL
# ============================================================


@st.cache_resource
def load_model():
    return joblib.load("notebooks/models/calibrated_model.pkl")


model = load_model()

# ============================================================
# LOAD FEATURES
# ============================================================

feature_names = joblib.load("notebooks/models/feature_names.pkl")

# ============================================================
# TITLE
# ============================================================

st.title("Trustworthy Diabetes Risk Prediction Framework")

st.markdown("""
### SHAP-Guided Medically Constrained Explainable AI Framework

This research framework integrates:

- Bayesian Optimization
- Calibrated CatBoost Modeling
- SHAP Explainability
- Medically Feasible Recourse Analysis
- Trustworthy Healthcare AI

---
""")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Project Overview",
        "Research Questions",
        "Live Prediction",
        "SHAP Explainability",
        "Recourse Simulation",
        "Results Summary",
        "Conclusion",
    ],
)

# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "Project Overview":

    st.header("Research Motivation")

    st.write("""
Traditional healthcare AI systems often lack transparency,
calibration reliability, and actionable interpretability.

This project proposes a trustworthy healthcare AI framework
integrating explainability and medically feasible recourse
for diabetes risk prediction.
""")

    st.subheader("Proposed Framework")

    framework = Image.open("results/figures/Framework.png")

    st.image(framework, width=700)

    st.subheader("Core Contributions")

    st.markdown("""
### Contributions

1. Bayesian-optimized diabetes prediction
2. Calibration-aware probability estimation
3. SHAP-based global and local explainability
4. Medically constrained recourse simulation
5. Actionable trustworthy healthcare AI framework
""")

# ============================================================
# PAGE 2 — RESEARCH QUESTIONS
# ============================================================

elif page == "Research Questions":

    st.header("Research Questions")

    # ========================================================

    st.subheader(
        "RQ1 — Can Bayesian optimization improve diabetes prediction performance?"
    )

    rq1 = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Random Forest",
                "XGBoost",
                "CatBoost",
                "Optimized CatBoost",
            ],
            "ROC-AUC": [0.837, 0.830, 0.802, 0.839, 0.848],
        }
    )

    st.dataframe(rq1, width=700)

    st.success("Optimized CatBoost achieved highest ROC-AUC performance.")

    st.markdown("---")

    # ========================================================

    st.subheader("RQ2 — Which clinical variables dominate diabetes risk prediction?")

    shap_summary = Image.open("results/figures/shap_summary.png")

    st.image(shap_summary, caption="Global SHAP Summary Plot", width=700)

    st.info("Glucose and BMI emerged as dominant clinically actionable variables.")

    st.markdown("---")

    # ========================================================

    st.subheader("RQ3 — Can medically feasible recourse reduce diabetes risk?")

    risk_reduction = Image.open("results/figures/risk_reduction.png")

    st.image(
        risk_reduction,
        caption="Risk Reduction Through Feasible Recourse",
        width=500,
    )

    st.success(
        "Risk probability reduced from 0.684 to 0.265 using feasible modifications."
    )

# ============================================================
# PAGE 3 — LIVE PREDICTION
# ============================================================

elif page == " Prediction":

    st.header("Diabetes Risk Prediction")

    col1, col2 = st.columns(2)

    with col1:

        pregnancies = st.slider("Pregnancies", 0, 17, 2)

        glucose = st.slider("Glucose", 50, 200, 120)

        bp = st.slider("Blood Pressure", 30, 130, 70)

        skin = st.slider("Skin Thickness", 0, 99, 20)

    with col2:

        insulin = st.slider("Insulin", 0, 850, 80)

        bmi = st.slider("BMI", 10.0, 50.0, 30.0)

        dpf = st.slider("Diabetes Pedigree Function", 0.05, 2.5, 0.5)

        age = st.slider("Age", 18, 90, 35)

    input_df = pd.DataFrame(
        [
            {
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": bp,
                "SkinThickness": skin,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": dpf,
                "Age": age,
            }
        ]
    )

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"High Diabetes Risk Probability: {probability:.3f}")
    else:
        st.success(f"Lower Diabetes Risk Probability: {probability:.3f}")

# ============================================================
# PAGE 4 — SHAP EXPLAINABILITY
# ============================================================

elif page == "SHAP Explainability":

    st.header("SHAP Explainability")

    st.subheader("Global Explainability")

    shap_summary = Image.open("results/figures/shap_summary.png")

    st.image(shap_summary, width=700)

    st.markdown("""
### Key Findings

- Glucose strongly increased diabetes risk
- BMI exhibited nonlinear influence
- Age moderately contributed toward risk
- Insulin importance affected by missingness uncertainty
""")

    st.subheader("Local Explainability")

    waterfall = Image.open("results/figures/waterfall_plot.png")

    st.image(waterfall, width=700)

# ============================================================
# PAGE 5 — RECOURSE SIMULATION
# ============================================================

elif page == "Recourse Simulation":

    st.header("Medically Feasible Recourse Simulation")

    st.write("""
Modify clinically actionable variables to observe
changes in calibrated diabetes risk probability.
""")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original High-Risk Patient")

        glucose_cf = st.slider("Modified Glucose", 80, 200, 130)

        bmi_cf = st.slider("Modified BMI", 15.0, 40.0, 24.0)

    query_instance = pd.DataFrame(
        [
            {
                "Pregnancies": 7,
                "Glucose": glucose_cf,
                "BloodPressure": 64,
                "SkinThickness": 29,
                "Insulin": 125,
                "BMI": bmi_cf,
                "DiabetesPedigreeFunction": 0.294,
                "Age": 40,
            }
        ]
    )

    cf_prob = model.predict_proba(query_instance)[0][1]

    with col2:

        st.subheader("Risk Outcome")

        st.metric("Predicted Diabetes Risk", round(cf_prob, 3))

        if cf_prob < 0.4:
            st.success("Lower-risk region")
        elif cf_prob < 0.6:
            st.warning("Moderate-risk region")
        else:
            st.error("High-risk region")

    st.subheader("Risk Reduction Visualization")

    risk_df = pd.DataFrame(
        {"Scenario": ["Original", "Modified"], "Risk": [0.684, cf_prob]}
    )

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.bar(risk_df["Scenario"], risk_df["Risk"])

    ax.set_ylabel("Predicted Risk")

    st.pyplot(fig)

# ============================================================
# PAGE 6 — RESULTS SUMMARY
# ============================================================

elif page == "Results Summary":

    st.header("Final Experimental Results")

    results_df = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Random Forest",
                "XGBoost",
                "CatBoost",
                "Optimized CatBoost",
            ],
            "Accuracy": [0.765, 0.765, 0.746, 0.763, 0.765],
            "F1-Score": [0.617, 0.642, 0.630, 0.640, 0.625],
            "ROC-AUC": [0.837, 0.830, 0.802, 0.839, 0.848],
        }
    )

    st.dataframe(results_df, width=700)

    calibration = Image.open("results/figures/calibration_curve.png")

    st.image(calibration, caption="Calibration Curve", width=500)

# ============================================================
# PAGE 7 — CONCLUSION
# ============================================================

elif page == "Conclusion":

    st.header(" Conclusion")

    st.markdown("""

### Final Findings

- Bayesian optimization improved discrimination capability
- Calibration improved probability reliability
- SHAP improved interpretability and transparency
- Glucose and BMI emerged as dominant actionable variables
- Feasible recourse substantially reduced predicted diabetes risk

---

### Final Takeaway

This framework demonstrates how explainability,
calibration, and medically feasible recourse
can improve trustworthiness in healthcare AI systems.
""")
