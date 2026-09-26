# Customer Churn Prediction ML System

An end-to-end Machine Learning project that predicts whether a customer is likely to churn based on customer-level behavioral and demographic features.

The project demonstrates a complete ML workflow including data generation, preprocessing, train-test splitting, feature scaling, Logistic Regression, model evaluation, cross-validation, model serialization, and inference on new customers.

---

## Problem Statement

Customer churn is an important business problem for subscription-based and service-oriented companies.

The objective of this project is to build a Machine Learning model that predicts whether a customer is likely to churn using:

- Age
- Monthly Spend
- Tenure
- Support Calls

The model produces both:

- Churn prediction
- Churn probability

---

## Dataset

This project uses a synthetic dataset containing 1,000 customer records.

### Features

| Feature | Description |
|---|---|
| age | Customer age |
| monthly_spend | Customer monthly spending |
| tenure_months | Number of months the customer has been with the company |
| support_calls | Number of customer support calls |
| churn | Target variable: 1 = Churn, 0 = No Churn |

### Dataset Distribution

- Total customers: 1,000
- No Churn: 690
- Churn: 310

The dataset is generated programmatically using NumPy and Pandas.

---

## Machine Learning Workflow

```text
Synthetic Data Generation
        ↓
Feature / Target Separation
        ↓
Train-Test Split
        ↓
StandardScaler
        ↓
Logistic Regression
        ↓
Model Evaluation
        ↓
Cross-Validation
        ↓
Model Serialization
        ↓
Prediction on New Customers

## Project Status

Portfolio project completed and deployed to GitHub.

## Model Details

The project uses Logistic Regression with StandardScaler inside a scikit-learn Pipeline.

The model predicts:
- Churn (1)
- No Churn (0)

The trained pipeline is saved using Joblib.