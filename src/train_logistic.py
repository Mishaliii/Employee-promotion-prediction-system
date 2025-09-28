"""
Train Logistic Regression model for Employee Promotion Prediction
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
import joblib
import os

# ---- Paths ----
DATA_PATH = "data/train.csv"
MODEL_PATH = "models/logreg_promotion_pipeline.pkl"
os.makedirs("models", exist_ok=True)

# ---- Load Data ----
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["is_promoted", "employee_id"])
y = df["is_promoted"]

# ---- Identify Columns ----
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# ---- Preprocessing ----
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# ---- Model ----
model = LogisticRegression(max_iter=2000, class_weight="balanced")

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", model)
])

# ---- Train/Validate Split ----
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Fit ----
pipeline.fit(X_train, y_train)

# ---- Evaluate ----
y_pred = pipeline.predict(X_val)
y_proba = pipeline.predict_proba(X_val)[:, 1]

print("Accuracy:", accuracy_score(y_val, y_pred))
print("Balanced Accuracy:", balanced_accuracy_score(y_val, y_pred))
print("ROC AUC:", roc_auc_score(y_val, y_proba))
print("\nClassification Report:\n", classification_report(y_val, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))

# ---- Save Model ----
os.makedirs("../models", exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)
print(f"\n✅ Model saved to {MODEL_PATH}")
