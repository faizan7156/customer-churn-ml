
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)


# ============================================================
# 1. Generate synthetic dataset
# ============================================================

np.random.seed(42)

n = 1000

age = np.random.randint(18, 70, n)
monthly_spend = np.random.randint(300, 2000, n)
tenure_months = np.random.randint(1, 61, n)
support_calls = np.random.randint(0, 10, n)


# Create synthetic churn score
churn_score = (
    0.08 * support_calls
    - 0.03 * tenure_months
    - 0.0002 * monthly_spend
    + 0.01 * age
)


# Convert score into binary target
churn = (churn_score > 0).astype(int)


df = pd.DataFrame({
    "age": age,
    "monthly_spend": monthly_spend,
    "tenure_months": tenure_months,
    "support_calls": support_calls,
    "churn": churn
})


print("Dataset shape:", df.shape)
print()
print(df.head())
print()

print("No churn (0):", (df["churn"] == 0).sum())
print("Churn (1):", (df["churn"] == 1).sum())


# ============================================================
# 2. Separate features and target
# ============================================================

X = df[
    [
        "age",
        "monthly_spend",
        "tenure_months",
        "support_calls"
    ]
]

y = df["churn"]


# ============================================================
# 3. Train-test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print()
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================================
# 4. Create ML Pipeline
# ============================================================

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression())
])


# ============================================================
# 5. Train model
# ============================================================

model.fit(X_train, y_train)


# ============================================================
# 6. Predictions
# ============================================================

y_pred = model.predict(X_test)

print()
print("Predictions:")
print(y_pred[:20])


# ============================================================
# 7. Classification metrics
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print()
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


# ============================================================
# 8. Confusion matrix
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print()
print("Confusion Matrix:")
print(cm)


# ============================================================
# 9. Prediction probabilities
# ============================================================

y_prob = model.predict_proba(X_test)

print()
print("First 10 probabilities:")
print(y_prob[:10])


# ============================================================
# 10. ROC-AUC
# ============================================================

auc = roc_auc_score(y_test, y_prob[:, 1])

print()
print("ROC-AUC:", auc)


# ============================================================
# 11. Model coefficients
# ============================================================

print()
print("Model coefficients:")

classifier = model.named_steps["classifier"]

for feature, coef in zip(X.columns, classifier.coef_[0]):
    print(feature, coef)

print("Intercept:", classifier.intercept_[0])

# ============================================================
# 12. 5-Fold Cross-Validation
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="f1"
)

print()
print("Cross-Validation F1 Scores:")
print(cv_scores)

print("Mean CV F1:", cv_scores.mean())

# ============================================================
# 13. Save trained model
# ============================================================

joblib.dump(model, "churn_model.pkl")

print()
print("Model saved as churn_model.pkl")