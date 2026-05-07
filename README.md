# Trustworthy Diabetes Risk Prediction using SHAP-Guided Medically Constrained Recourse

A research-oriented healthcare Explainable AI (XAI) framework integrating predictive modeling, calibration, SHAP explainability, and medically feasible recourse analysis for interpretable diabetes risk prediction.

---

## Overview

This project proposes a trustworthy healthcare AI pipeline that combines:

- Machine Learning-based diabetes risk prediction
- Bayesian hyperparameter optimization
- Probability calibration
- SHAP-based explainability
- Medically constrained actionable recourse analysis

The framework focuses on improving:

- interpretability,
- transparency,
- probabilistic reliability,
- and actionable decision support in healthcare machine learning systems.

---

## Framework Pipeline

```text
Pima Indians Diabetes Dataset
        ↓
Preprocessing & Missing Value Handling
        ↓
Baseline ML Models
        ↓
Bayesian Hyperparameter Optimization
        ↓
Calibrated CatBoost Model
        ↓
SHAP Explainability Analysis
        ↓
Actionable Feature Selection
        ↓
Medically Feasible Recourse Simulation
        ↓
Risk Reduction Evaluation
```

## Dataset

### Pima Indians Diabetes Dataset

The experiments were conducted using the publicly available Pima Indians Diabetes Dataset, which contains diagnostic healthcare measurements for diabetes risk prediction among female patients of Pima Indian heritage aged 21 years and above.

Dataset source:
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

### Dataset Features

| Feature                  | Description                                         |
| ------------------------ | --------------------------------------------------- |
| Pregnancies              | Number of pregnancies                               |
| Glucose                  | Plasma glucose concentration                        |
| BloodPressure            | Diastolic blood pressure (mm Hg)                    |
| SkinThickness            | Triceps skin fold thickness (mm)                    |
| Insulin                  | 2-Hour serum insulin (mu U/ml)                      |
| BMI                      | Body mass index                                     |
| DiabetesPedigreeFunction | Genetic diabetes likelihood score                   |
| Age                      | Age of patient                                      |
| Outcome                  | Diabetes diagnosis (0 = Non-diabetic, 1 = Diabetic) |

## Methodology

The proposed framework integrates predictive modeling, Bayesian optimization, calibration analysis, SHAP-based explainability, and medically constrained recourse simulation for trustworthy diabetes risk prediction.

### Workflow

1. Data preprocessing and missing value handling
2. Baseline machine learning model training
3. Bayesian hyperparameter optimization
4. Probability calibration
5. SHAP explainability analysis
6. Actionable feature identification
7. Medically feasible recourse simulation
8. Risk reduction evaluation

### Models Used

- Logistic Regression
- Random Forest
- XGBoost
- CatBoost

### Explainability

SHAP (SHapley Additive exPlanations) was utilized for:

- global feature importance analysis,
- local prediction interpretation,
- and nonlinear feature contribution analysis.

The explainability framework identified:

- Glucose
- BMI

as dominant actionable contributors toward diabetes risk prediction.

### Medically Constrained Recourse

The framework performs feasible recourse simulation by modifying only clinically actionable variables while preserving immutable patient attributes.

For representative high-risk patient instances, medically plausible feature modifications substantially reduced predicted diabetes risk probabilities.

---

## Key Features

- Bayesian-optimized CatBoost classifier
- Probability calibration analysis
- SHAP-based global and local explainability
- Actionable feature selection
- Medically feasible recourse generation
- Risk reduction evaluation
- Trustworthy healthcare AI workflow

---

## Experimental Highlights

| Metric   | Optimized CatBoost |
| -------- | ------------------ |
| Accuracy | 0.765              |
| ROC-AUC  | 0.848              |
| F1-Score | 0.625              |

### Key Findings

- Glucose and BMI emerged as dominant predictive features.
- SHAP explanations aligned with established clinical diabetes risk factors.
- Medically feasible recourse simulation reduced diabetes risk probability from:

```text
0.684 → 0.265
```
