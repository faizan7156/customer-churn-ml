import joblib
import pandas as pd

# Load trained model
model = joblib.load("churn_model.pkl")

# Test customers
customers = pd.DataFrame([
    {
        "age": 32,
        "monthly_spend": 1954,
        "tenure_months": 15,
        "support_calls": 9
    },
    {
        "age": 45,
        "monthly_spend": 1500,
        "tenure_months": 50,
        "support_calls": 1
    },
    {
        "age": 25,
        "monthly_spend": 500,
        "tenure_months": 3,
        "support_calls": 8
    }
])

# Predictions
predictions = model.predict(customers)
probabilities = model.predict_proba(customers)[:, 1]

# Display results
for i in range(len(customers)):

    print(f"\nCustomer {i + 1}")
    print("-" * 30)

    print("Age:", customers.iloc[i]["age"])
    print("Monthly Spend:", customers.iloc[i]["monthly_spend"])
    print("Tenure:", customers.iloc[i]["tenure_months"])
    print("Support Calls:", customers.iloc[i]["support_calls"])

    print(
        "Prediction:",
        "Churn" if predictions[i] == 1 else "No Churn"
    )

    print(f"Churn Probability: {probabilities[i]:.2%}")