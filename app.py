import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from PIL import Image

import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(page_title="Trustworthy Diabetes XAI", layout="wide")

# ============================================================
# LOAD MODEL
# ============================================================


@st.cache_resource
def load_model():
    return joblib.load("notebooks/models/calibrated_model.pkl")


model = load_model()

# ============================================================
# LOAD DATASET
# ============================================================

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]


@st.cache_data
def load_data():

    df = pd.read_csv(url, names=columns)

    return df


df = load_data()
# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Research Navigation")

page = st.sidebar.radio(
    "Research Objectives",
    [
        "Objective 1 — Diabetes Risk Prediction",
        "Objective 2 — SHAP Explainability",
        "Objective 3 — Medically Feasible Recourse",
        "Final Results and Conclusion",
    ],
)

# ============================================================
# OBJECTIVE 1
# ============================================================

if page == "Objective 1 — Diabetes Risk Prediction":

    st.title("Objective 1 — Diabetes Risk Prediction")

    st.markdown("""
This objective focuses on building a trustworthy machine learning
framework capable of predicting diabetes risk using clinical
healthcare features.

The module below demonstrates:
- clinical feature understanding,
- feature engineering,
- machine learning prediction,
- interactive dataset exploration,
- and live diabetes risk prediction.
""")

    st.markdown("---")

    # ========================================================
    # CLINICAL FEATURES
    # ========================================================

    st.header("Clinical Features Used")

    feature_df = pd.DataFrame(
        {
            "Feature": [
                "Glucose",
                "BMI",
                "Blood Pressure",
                "Insulin",
                "Age",
                "Pregnancies",
                "Skin Thickness",
                "Diabetes Pedigree Function",
            ],
            "Clinical Importance": [
                "Primary diabetes indicator",
                "Obesity and metabolic health",
                "Cardiovascular relationship",
                "Insulin resistance estimation",
                "Age-related diabetes risk",
                "Pregnancy-associated diabetes patterns",
                "Body fat distribution",
                "Genetic predisposition",
            ],
        }
    )

    st.dataframe(feature_df, use_container_width=True)

    st.info("""
These physiological and hereditary features collectively help
machine learning models identify hidden diabetes risk patterns.
""")

    st.markdown("---")

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    st.header("Feature Engineering")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Data Cleaning")

        st.markdown("""
- Invalid clinical zero values detected
- Missing observations identified
- Median imputation applied
- Robust preprocessing pipeline
""")

    with col2:

        st.subheader("Feature Transformation")

        st.markdown("""
- Feature standardization performed
- Stable optimization space created
- Improved model convergence
- Enhanced feature comparability
""")

    st.markdown("---")

    # ========================================================
    # PIPELINE
    # ========================================================

    st.header("Machine Learning Pipeline")

    st.code("""

Clinical Patient Data
        ↓
Feature Engineering
        ↓
Optimized CatBoost Model
        ↓
Probability Calibration
        ↓
Diabetes Risk Prediction

""")

    st.markdown("---")

    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.header("Optimized Model Performance")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric("Accuracy", "76.5%")

    with metric2:
        st.metric("F1-Score", "0.625")

    with metric3:
        st.metric("ROC-AUC", "0.848")

    comparison_df = pd.DataFrame(
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

    fig = px.bar(
        comparison_df,
        x="Model",
        y="ROC-AUC",
        text="ROC-AUC",
        title="Model Performance Comparison",
    )

    fig.update_layout(height=450)

    st.plotly_chart(fig, use_container_width=True)

    st.success("""
Bayesian optimization improved predictive discrimination
and produced a reliable calibrated diabetes prediction model.
""")

    st.markdown("---")

    # ========================================================
    # INTERACTIVE DATA EXPLORATION
    # ========================================================

    st.header("Clinical Features")

    st.markdown("""
Explore how clinical variables behave across diabetic and
non-diabetic patient populations.
""")

    # LOAD DATASET

    # df = pd.read_csv("data/diabetes.csv")

    selected_feature = st.selectbox(
        "Select Clinical Feature",
        ["Glucose", "BMI", "Age", "Insulin", "BloodPressure", "SkinThickness"],
    )

    # ========================================================
    # HISTOGRAM
    # ========================================================

    hist_fig = px.histogram(
        df,
        x=selected_feature,
        color="Outcome",
        marginal="box",
        barmode="overlay",
        title=f"{selected_feature} Distribution by Diabetes Outcome",
    )

    hist_fig.update_layout(height=500)

    st.plotly_chart(hist_fig, use_container_width=True)

    # ========================================================
    # INTERPRETATION
    # ========================================================

    if selected_feature == "Glucose":

        st.info("""
Patients with diabetes generally exhibit higher glucose levels,
making glucose one of the strongest predictive variables.
""")

    elif selected_feature == "BMI":

        st.info("""
Higher BMI values are associated with obesity-related metabolic
risk and contribute strongly toward diabetes prediction.
""")

    elif selected_feature == "Age":

        st.info("""
Diabetes risk generally increases with age due to long-term
metabolic and physiological changes.
""")

    # ========================================================
    # FEATURE RELATIONSHIP
    # ========================================================

    st.markdown("---")

    st.subheader("Feature Relationship")

    x_feature = st.selectbox(
        "Select X-axis Feature", ["Glucose", "BMI", "Age", "Insulin"], key="x_axis"
    )

    y_feature = st.selectbox(
        "Select Y-axis Feature", ["BMI", "Glucose", "Age", "Insulin"], key="y_axis"
    )

    scatter_fig = px.scatter(
        df,
        x=x_feature,
        y=y_feature,
        color="Outcome",
        opacity=0.7,
        title=f"{x_feature} vs {y_feature}",
    )

    scatter_fig.update_layout(height=550)

    st.plotly_chart(scatter_fig, use_container_width=True)

    st.success("""
Machine learning models learn hidden multidimensional relationships
between clinical variables to identify diabetes risk patterns.
""")

    st.markdown("---")

    # ========================================================
    # LIVE PREDICTION
    # ========================================================

    st.header("Diabetes Risk Prediction Demo")

    st.markdown("""
Modify patient clinical variables below to observe how
machine learning predicts diabetes risk probability.
""")

    col1, col2 = st.columns(2)

    with col1:

        glucose = st.slider("Glucose", 50, 200, 120, key="obj1_glucose")

        bmi = st.slider("BMI", 10.0, 50.0, 30.0, key="obj1_bmi")

        age = st.slider("Age", 18, 90, 35, key="obj1_age")

        insulin = st.slider("Insulin", 0, 850, 80, key="obj1_insulin")

    with col2:

        pregnancies = st.slider("Pregnancies", 0, 17, 2, key="obj1_preg")

        bp = st.slider("Blood Pressure", 30, 130, 70, key="obj1_bp")

        skin = st.slider("Skin Thickness", 0, 99, 20, key="obj1_skin")

    # ========================================================
    # PREDICTION INPUT
    # ========================================================

    input_df = pd.DataFrame(
        [
            {
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": bp,
                "SkinThickness": skin,
                "Insulin": insulin,
                "BMI": bmi,
                "Age": age,
            }
        ]
    )

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")

    # ========================================================
    # OUTPUT
    # ========================================================

    st.header("Prediction Output")

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:

        st.metric("Predicted Diabetes Risk", f"{probability:.3f}")

    with metric_col2:

        if prediction == 1:
            st.error("High Diabetes Risk")

        else:
            st.success("Lower Diabetes Risk")

    # ========================================================
    # DYNAMIC RISK GAUGE
    # ========================================================

    st.subheader("Risk Gauge")

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={"text": "Diabetes Risk Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.3},
                "steps": [
                    {"range": [0, 40], "color": "lightgreen"},
                    {"range": [40, 70], "color": "gold"},
                    {"range": [70, 100], "color": "salmon"},
                ],
            },
        )
    )

    gauge.update_layout(height=350)

    st.plotly_chart(gauge, use_container_width=True)

    # ========================================================
    # FEATURE PROFILE
    # ========================================================

    st.subheader("Patient Clinical Profile")

    profile_df = pd.DataFrame(
        {
            "Feature": ["Glucose", "BMI", "Age", "Insulin", "Blood Pressure"],
            "Value": [glucose, bmi, age, insulin, bp],
        }
    )

    profile_fig = px.bar(
        profile_df,
        x="Value",
        y="Feature",
        orientation="h",
        title="Clinical Feature Profile",
    )

    profile_fig.update_layout(height=400)

    st.plotly_chart(profile_fig, use_container_width=True)

    st.markdown("---")

    # ========================================================
    # WHY SHAP
    # ========================================================

    st.header("Why is Explainability Needed?")

    st.warning("""
Although the optimized CatBoost model predicts diabetes risk effectively,
the prediction process still behaves like a black-box system.

Clinicians may ask:

- Why was this patient classified as high-risk?
- Which clinical features contributed most?
- Which variables increased or reduced risk?
- How can the risk be reduced safely?

This motivates Objective 2:
SHAP-based Explainable AI for transparent healthcare prediction.
""")

    st.info("""
Objective 2 focuses on interpreting prediction behavior using
SHAP explainability to identify clinically influential variables.
""")


# ============================================================
# OBJECTIVE 2
# ============================================================

elif page == "Objective 2 — SHAP Explainability":

    st.title("Objective 2 — SHAP Explainability")

    st.markdown("""
This objective focuses on explaining how the optimized
machine learning model makes diabetes predictions using
SHAP (SHapley Additive exPlanations).

The goal is to transform the black-box prediction process
into a transparent and clinically interpretable system.
""")

    st.markdown("---")

    # ========================================================
    # WHY EXPLAINABILITY
    # ========================================================

    st.header("Why Explainability is Needed")

    col1, col2 = st.columns(2)

    with col1:

        st.error("""
Traditional black-box machine learning models only provide:
- prediction labels,
- probability scores,
without explaining WHY the prediction occurred.
""")

    with col2:

        st.success("""
SHAP explains:
- which clinical features contributed most,
- whether features increased or decreased risk,
- and how each prediction was formed.
""")

    st.markdown("---")

    # ========================================================
    # SHAP INTRODUCTION
    # ========================================================

    st.header("Understanding SHAP")

    st.markdown("""
SHAP assigns contribution scores to each feature based on
cooperative game theory principles.

Each feature receives a SHAP value representing:
- positive contribution toward diabetes risk,
- or negative contribution reducing risk.
""")

    st.info("""
Positive SHAP values increase diabetes risk prediction,
while negative SHAP values reduce predicted risk.
""")

    st.markdown("---")

    # ========================================================
    # GLOBAL EXPLAINABILITY
    # ========================================================

    st.header("Global Explainability")

    st.markdown("""
Global explainability helps understand how the model behaves
across the entire patient population.
""")

    shap_summary = Image.open("results/figures/shap_summary.png")

    col1, col2, col3 = st.columns([1, 4, 1])

    with col2:

        st.image(shap_summary, width=750)

    st.success("""
The SHAP summary plot reveals that:
- Glucose is the most influential predictor,
- BMI strongly impacts diabetes prediction,
- Age and Insulin also contribute significantly.
""")

    st.markdown("---")

    # ========================================================
    # INTERACTIVE FEATURE IMPORTANCE
    # ========================================================

    st.header("Interactive Feature Importance")

    importance_df = pd.DataFrame(
        {
            "Feature": [
                "Glucose",
                "BMI",
                "Age",
                "Insulin",
                "Pregnancies",
                "BloodPressure",
                "SkinThickness",
                "DiabetesPedigreeFunction",
            ],
            "Importance": [0.32, 0.21, 0.13, 0.10, 0.08, 0.06, 0.05, 0.05],
        }
    )

    importance_fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        title="Global SHAP Feature Importance",
    )

    importance_fig.update_layout(height=500)

    st.plotly_chart(importance_fig, use_container_width=True)

    st.markdown("""
The model primarily relies on glucose and BMI because these
features exhibit strong separability between diabetic and
non-diabetic patient populations.
""")

    st.markdown("---")

    # ========================================================
    # FEATURE DEPENDENCE
    # ========================================================

    st.header("SHAP Dependence Analysis")

    st.markdown("""
Dependence plots show how changing a clinical feature affects
predicted diabetes risk.
""")

    feature_choice = st.selectbox(
        "Select Feature for Dependence Analysis", ["Glucose", "BMI"]
    )

    if feature_choice == "Glucose":

        dep_img = Image.open("results/figures/shap_dependence_glucose.png")

        st.image(dep_img, width=700)

        st.info("""
As glucose levels increase, SHAP values become increasingly positive,
indicating higher diabetes risk contribution.
""")

    elif feature_choice == "BMI":

        dep_img = Image.open("results/figures/shap_dependence_BMI.png")

        st.image(dep_img, width=700)

        st.info("""
Higher BMI values generally increase diabetes risk contribution,
reflecting obesity-associated metabolic risk.
""")

    st.markdown("---")

    # ========================================================
    # LOCAL EXPLAINABILITY
    # ========================================================

    st.header("Local Patient-Level Explainability")

    st.markdown("""
Local explainability focuses on understanding predictions
for an individual patient.
""")

    waterfall = Image.open("results/figures/waterfall_plot.png")

    col1, col2, col3 = st.columns([1, 4, 1])

    with col2:

        st.image(waterfall, width=650)

    st.success("""
The SHAP waterfall plot explains how individual features
collectively pushed the patient toward high diabetes risk.
""")

    st.markdown("---")

    # ========================================================
    # INTERACTIVE PATIENT EXPLANATION
    # ========================================================

    st.header("Interactive Clinical Interpretation")

    glucose_level = st.slider("Glucose Level", 50, 200, 140, key="shap_glucose")

    bmi_level = st.slider("BMI Level", 10.0, 50.0, 32.0, key="shap_bmi")

    shap_demo_df = pd.DataFrame(
        {
            "Feature": ["Glucose", "BMI"],
            "Contribution": [glucose_level / 200, bmi_level / 50],
        }
    )

    contrib_fig = px.bar(
        shap_demo_df,
        x="Feature",
        y="Contribution",
        color="Contribution",
        title="Illustrative Feature Contribution",
    )

    contrib_fig.update_layout(height=400)

    st.plotly_chart(contrib_fig, use_container_width=True)

    st.info("""
This simplified visualization demonstrates how increasing
clinical measurements can increase model contribution toward
diabetes prediction.
""")

    st.markdown("---")

    # ========================================================
    # BLACK BOX TO TRANSPARENCY
    # ========================================================

    st.header("From Black-Box AI to Transparent AI")

    comparison_col1, comparison_col2 = st.columns(2)

    with comparison_col1:

        st.error("""
Without Explainability:
- Hidden prediction reasoning
- Low clinician trust
- Difficult medical interpretation
- Poor transparency
""")

    with comparison_col2:

        st.success("""
With SHAP Explainability:
- Transparent predictions
- Clinically interpretable reasoning
- Feature-level understanding
- Improved trustworthy AI
""")

    st.markdown("---")

    # ========================================================
    # TRANSITION TO OBJECTIVE 3
    # ========================================================

    st.header("Transition to Objective 3")

    st.warning("""
SHAP explains WHY a patient is high-risk.

However, clinicians still need to know:

- Which features can be modified?
- How can risk be reduced safely?
- What actionable recommendations exist?

This motivates Objective 3:
Medically Feasible Counterfactual Recourse.
""")


# ============================================================
# OBJECTIVE 3
# ============================================================

elif page == "Objective 3 — Medically Feasible Recourse":

    st.title("Objective 3 — Generate Counterfactual Explanations")

    st.markdown("""
This objective focuses on generating medically feasible
counterfactual explanations to reduce predicted diabetes risk.

While SHAP explains WHY a patient is high-risk,
counterfactual recourse explains:

- how the risk can be reduced,
- which features should change,
- and which recommendations are clinically actionable.
""")

    st.markdown("---")

    # ========================================================
    # WHAT ARE COUNTERFACTUALS
    # ========================================================

    st.header("What are Counterfactual Explanations?")

    col1, col2 = st.columns(2)

    with col1:

        st.error("""
Traditional explainability methods only explain:
WHY the prediction occurred.
""")

    with col2:

        st.success("""
Counterfactual explanations answer:
'What should change to reduce risk?'
""")

    st.info("""
Counterfactual explanations generate minimally modified
patient profiles that shift prediction outcomes from
high-risk to lower-risk regions.
""")

    st.markdown("---")

    # ========================================================
    # MEDICAL FEASIBILITY
    # ========================================================

    st.header("Medically Feasible Feature Constraints")

    st.markdown("""
Not all clinical variables should be modified.

This framework separates:
- actionable features,
- and immutable features.
""")

    feature_type_df = pd.DataFrame(
        {
            "Feature Type": ["Actionable Features", "Immutable Features"],
            "Examples": ["Glucose, BMI, Insulin", "Age, Genetics, Pregnancies"],
            "Clinical Meaning": [
                "Can be improved through intervention",
                "Cannot realistically be changed",
            ],
        }
    )

    st.dataframe(feature_type_df, use_container_width=True)

    st.warning("""
Only medically actionable variables are modified during
counterfactual generation to ensure realistic healthcare recommendations.
""")

    st.markdown("---")

    # ========================================================
    # ORIGINAL HIGH-RISK PATIENT
    # ========================================================

    st.header("Original High-Risk Patient")

    original_patient = pd.DataFrame(
        {
            "Feature": [
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age",
            ],
            "Original Value": [7, 159, 64, 29, 125, 27.4, 0.294, 40],
        }
    )

    st.dataframe(original_patient, use_container_width=True)

    st.error("""
Original calibrated diabetes risk probability: 0.684
""")

    st.markdown("---")

    # ========================================================
    # INTERACTIVE COUNTERFACTUAL SIMULATION
    # ========================================================

    st.header("Counterfactual Simulation")

    st.markdown("""
Modify clinically actionable features below and observe
how diabetes risk probability changes dynamically.
""")

    col1, col2 = st.columns(2)

    with col1:

        glucose_cf = st.slider("Modified Glucose", 80, 200, 130, key="cf_glucose")

        bmi_cf = st.slider("Modified BMI", 15.0, 45.0, 24.0, key="cf_bmi")

    with col2:

        insulin_cf = st.slider("Modified Insulin", 0, 300, 110, key="cf_insulin")

        bp_cf = st.slider("Modified Blood Pressure", 40, 120, 70, key="cf_bp")

    # ========================================================
    # COUNTERFACTUAL INPUT
    # ========================================================

    counterfactual_input = pd.DataFrame(
        [
            {
                "Pregnancies": 7,
                "Glucose": glucose_cf,
                "BloodPressure": bp_cf,
                "SkinThickness": 29,
                "Insulin": insulin_cf,
                "BMI": bmi_cf,
                "DiabetesPedigreeFunction": 0.294,
                "Age": 40,
            }
        ]
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    cf_probability = model.predict_proba(counterfactual_input)[0][1]

    st.markdown("---")

    # ========================================================
    # BEFORE VS AFTER
    # ========================================================

    st.header("Counterfactual Risk Comparison")

    comparison_col1, comparison_col2 = st.columns(2)

    with comparison_col1:

        st.metric("Original Risk", "0.684")

    with comparison_col2:

        st.metric("Counterfactual Risk", f"{cf_probability:.3f}")

    # ========================================================
    # DYNAMIC RISK VISUALIZATION
    # ========================================================

    risk_df = pd.DataFrame(
        {"Scenario": ["Original", "Counterfactual"], "Risk": [0.684, cf_probability]}
    )

    risk_fig = px.bar(
        risk_df,
        x="Scenario",
        y="Risk",
        color="Scenario",
        text="Risk",
        title="Risk Reduction Through Counterfactual Recourse",
    )

    risk_fig.update_layout(height=450)

    st.plotly_chart(risk_fig, use_container_width=True)

    # ========================================================
    # INTERPRETATION
    # ========================================================

    if cf_probability < 0.4:

        st.success("""
The modified patient profile successfully reduced
predicted diabetes risk into a lower-risk region.
""")

    elif cf_probability < 0.6:

        st.warning("""
The modified patient profile reduced risk partially,
but further clinical improvement may still be required.
""")

    else:

        st.error("""
Risk remains relatively high despite modifications,
indicating additional intervention may be necessary.
""")

    st.markdown("---")

    # ========================================================
    # FEATURE CHANGE ANALYSIS
    # ========================================================

    st.header("Actionable Feature Modification Analysis")

    change_df = pd.DataFrame(
        {
            "Feature": ["Glucose", "BMI", "Insulin", "Blood Pressure"],
            "Original": [159, 27.4, 125, 64],
            "Modified": [glucose_cf, bmi_cf, insulin_cf, bp_cf],
        }
    )

    melted_df = change_df.melt(
        id_vars="Feature",
        value_vars=["Original", "Modified"],
        var_name="Scenario",
        value_name="Value",
    )

    change_fig = px.bar(
        melted_df,
        x="Feature",
        y="Value",
        color="Scenario",
        barmode="group",
        title="Original vs Counterfactual Features",
    )

    change_fig.update_layout(height=500)

    st.plotly_chart(change_fig, use_container_width=True)

    st.info("""
Counterfactual generation attempts to minimize changes
while still reducing predicted diabetes risk.
""")

    st.markdown("---")

    # ========================================================
    # COUNTERFACTUAL LOGIC
    # ========================================================

    st.header("Counterfactual Generation Logic")

    st.code("""

Original High-Risk Patient
        ↓
Identify Actionable Features
        ↓
Modify Feasible Clinical Variables
        ↓
Recompute Prediction Probability
        ↓
Generate Lower-Risk Counterfactual

""")

    st.markdown("---")

    # ========================================================
    # TRUSTWORTHY AI
    # ========================================================

    st.header("From Explainability to Actionable AI")

    col1, col2 = st.columns(2)

    with col1:

        st.error("""
Without Counterfactual Recourse:
- Model only predicts risk
- No actionable guidance
- Limited clinical usability
""")

    with col2:

        st.success("""
With Counterfactual Recourse:
- Actionable recommendations
- Clinically feasible interventions
- Human-centered trustworthy AI
""")

    st.markdown("---")

    # ========================================================
    # FINAL TAKEAWAY
    # ========================================================

    st.header("Objective 3 Takeaway")

    st.success("""
Counterfactual explanations transform predictive healthcare AI
into actionable healthcare intelligence by identifying realistic
clinical modifications capable of reducing predicted diabetes risk.
""")
# ============================================================
# FINAL RESULTS AND CONCLUSION
# ============================================================

elif page == "Final Results and Conclusion":

    st.title("Research Questions, Results and Conclusion")

    st.markdown("""
This section summarizes how the proposed framework addressed
the three primary research questions of this study.

The final system integrates:
- optimized diabetes prediction,
- SHAP explainability,
- and medically feasible counterfactual recourse
to improve trustworthy healthcare AI.
""")

    st.markdown("---")

    # ========================================================
    # RESEARCH QUESTION 1
    # ========================================================

    st.header("Research Question 1")

    st.subheader("""
Can machine learning effectively predict diabetes risk using clinical healthcare features?
""")

    rq1_col1, rq1_col2 = st.columns([1, 1])

    with rq1_col1:

        st.success("""
### Findings

- Optimized CatBoost achieved strongest ROC-AUC performance
- Bayesian optimization improved predictive capability
- Calibration improved probability reliability
- Clinical features successfully captured diabetes risk patterns
""")

    with rq1_col2:

        performance_df = pd.DataFrame(
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

        rq1_fig = px.bar(
            performance_df,
            x="Model",
            y="ROC-AUC",
            color="ROC-AUC",
            text="ROC-AUC",
            title="RQ1 — Model Performance Comparison",
        )

        rq1_fig.update_layout(height=450)

        st.plotly_chart(rq1_fig, use_container_width=True)

    st.info("""
Research Question 1 was successfully addressed through Bayesian-optimized diabetes risk prediction.
""")

    st.markdown("---")

    # ========================================================
    # RESEARCH QUESTION 2
    # ========================================================

    st.header("Research Question 2")

    st.subheader("""
Can SHAP explainability improve transparency and interpretability in diabetes prediction?
""")

    rq2_col1, rq2_col2 = st.columns([1, 1])

    with rq2_col1:

        shap_summary = Image.open("results/figures/shap_summary.png")

        st.image(shap_summary, width=650)

    with rq2_col2:

        st.success("""
### Findings

- SHAP identified glucose as dominant predictor
- BMI strongly influenced diabetes prediction
- Local explanations improved patient-level interpretation
- Global explanations improved model transparency
""")

        importance_df = pd.DataFrame(
            {
                "Feature": ["Glucose", "BMI", "Age", "Insulin", "Pregnancies"],
                "Importance": [0.32, 0.21, 0.13, 0.10, 0.08],
            }
        )

        rq2_fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            title="RQ2 — SHAP Feature Importance",
        )

        rq2_fig.update_layout(height=400)

        st.plotly_chart(rq2_fig, use_container_width=True)

    st.info("""
Research Question 2 was successfully addressed through SHAP-based global and local explainability.
""")

    st.markdown("---")

    # ========================================================
    # RESEARCH QUESTION 3
    # ========================================================

    st.header("Research Question 3")

    st.subheader("""
Can medically feasible counterfactual explanations reduce predicted diabetes risk?
""")

    rq3_col1, rq3_col2 = st.columns([1, 1])

    with rq3_col1:

        risk_df = pd.DataFrame(
            {"Scenario": ["Original", "Counterfactual"], "Risk": [0.684, 0.265]}
        )

        rq3_fig = px.bar(
            risk_df,
            x="Scenario",
            y="Risk",
            color="Scenario",
            text="Risk",
            title="RQ3 — Counterfactual Risk Reduction",
        )

        rq3_fig.update_layout(height=450)

        st.plotly_chart(rq3_fig, use_container_width=True)

    with rq3_col2:

        st.success("""
### Findings

- Counterfactual explanations generated actionable guidance
- Glucose and BMI emerged as key modifiable variables
- Risk probability reduced significantly
- Recommendations remained medically feasible
""")

        st.metric("Risk Reduction", "0.684 → 0.265")

    st.info("""
Research Question 3 was successfully addressed through medically feasible counterfactual recourse generation.
""")

    st.markdown("---")

    # ========================================================
    # FINAL SYSTEM INTEGRATION
    # ========================================================

    st.header("Final Integrated Framework")

    st.code("""

Clinical Healthcare Data
        ↓
Feature Engineering
        ↓
Optimized CatBoost Prediction
        ↓
Probability Calibration
        ↓
SHAP Explainability
        ↓
Counterfactual Recourse
        ↓
Trustworthy Healthcare AI

""")

    st.markdown("---")

    # ========================================================
    # CALIBRATION
    # ========================================================

    st.header(" Probability Calibration")

    calibration_img = Image.open("results/figures/calibration_curve.png")

    col1, col2, col3 = st.columns([1, 4, 1])

    with col2:

        st.image(calibration_img, width=600)

    st.success("""
Calibration improved probability reliability,
which is critical for trustworthy healthcare AI systems.
""")

    st.markdown("---")

    # ========================================================
    # KEY CONTRIBUTIONS
    # ========================================================

    st.header("Key Research Contributions")

    contribution_col1, contribution_col2 = st.columns(2)

    with contribution_col1:

        st.success("""
### Technical Contributions

- Bayesian optimization
- Calibrated CatBoost modeling
- SHAP explainability
- Counterfactual recourse generation
""")

    with contribution_col2:

        st.success("""
### Healthcare AI Contributions

- Transparent prediction reasoning
- Actionable healthcare recommendations
- Human-centered AI framework
- Improved trustworthy AI
""")

    st.markdown("---")

    # ========================================================
    # FINAL CONCLUSION
    # ========================================================

    st.header("Final Conclusion")

    st.markdown("""
The proposed framework successfully integrated:

- optimized diabetes risk prediction,
- interpretable SHAP explainability,
- and medically feasible counterfactual recourse

to create a trustworthy healthcare AI system.

The framework demonstrates how explainability and actionable
recourse can improve transparency, interpretability,
and clinical usefulness in machine learning-based
healthcare prediction systems.
""")

    st.markdown("---")

    # ========================================================
    # FUTURE WORK
    # ========================================================

    st.header(" Future Scope")

    future_df = pd.DataFrame(
        {
            "Future Direction": [
                "Real-world clinical deployment",
                "Multimodal healthcare data",
                "Personalized intervention systems",
                "Temporal diabetes progression modeling",
                "Federated healthcare learning",
            ]
        }
    )

    st.dataframe(future_df, use_container_width=True)

    st.info("""
Future work can extend this framework toward
real-time clinical decision support systems
and personalized healthcare intelligence.
""")
