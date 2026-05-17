# Customer Churn Prediction with CatBoost + SHAP

## Overview
Predicting customer churn for a telecom company using CatBoost,
with Optuna hyperparameter tuning and SHAP explainability.

## Live App
[Click here to try the app](your-streamlit-link-here)

## Dataset
Telco Customer Churn — Kaggle (7,043 rows, 21 features)

## Key EDA Findings
## Complete EDA Summary — 7 Key Findings

**Dataset:** 7,032 telecom customers, 20 features, 26.5% churn rate

### Finding 1 — Class Imbalance
- 73.5% No churn, 26.5% churn
- Accuracy alone is misleading — will use F1 and AUC-ROC
- Will apply SMOTE on training data to handle imbalance

### Finding 2 — Contract Type (Strongest Signal)
- Month-to-month: 42.7% churn
- One year: 11.3% churn
- Two year: 2.8% churn
- 15x difference between month-to-month and two-year customers

### Finding 3 — Tenure and Monthly Charges
- Churned customers: median tenure 10 months, median bill $79.65
- Retained customers: median tenure 38 months, median bill $64.45
- New + high-paying customers are the most at-risk

### Finding 4 — Highest Risk Segment
- Month-to-month + Fiber Optic + Electronic Check = 60.4% churn
- 2.3x higher than overall average
- 6 in 10 customers in this segment leave

### Finding 5 — Security Services as Retention Anchors
- Without OnlineSecurity: 41.8% churn
- With OnlineSecurity: 14.6% churn — 2.9x difference
- TechSupport shows same pattern: 41.6% vs 15.2%
- Streaming services show almost no difference

### Finding 6 — Senior Citizens
- Senior citizens churn at 41.7% vs 23.7% for non-seniors
- Senior + Fiber Optic = 47.3% churn — second highest risk segment

### Finding 7 — Gender (Negative Finding)
- Female: 27.0%, Male: 26.2% — 0.8% difference
- Gender has no predictive power
- Churn is driven by service choices, not demographics

### Multicollinearity Note
- TotalCharges vs tenure: 0.83 correlation
- TotalCharges vs MonthlyCharges: 0.65 correlation
- Will consider dropping TotalCharges in feature engineering

### What This Means for Modeling
- Features to watch: Contract, tenure, MonthlyCharges,
  InternetService, OnlineSecurity, TechSupport, PaymentMethod
- Features likely low importance: gender, PhoneService
- Need SMOTE for class imbalance
- Need to handle TotalCharges multicollinearity

## Model Results
(fill after Day 12)

## Tech Stack
Python · Pandas · CatBoost · Optuna · SHAP · WandB · Streamlit
