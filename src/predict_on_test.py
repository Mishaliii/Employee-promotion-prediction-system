import pandas as pd
import joblib

# ---- Paths ----
MODEL_PATH = "models/logreg_promotion_pipeline.pkl"
TEST_PATH = "data/test.csv"
OUTPUT_PATH = "models/submission_logreg.csv"

# ---- Load Model ----
model = joblib.load(MODEL_PATH)

# ---- Load Test Data ----
test_df = pd.read_csv(TEST_PATH)

# Save employee_id
emp_id = test_df["employee_id"]

# Drop id column
X_test = test_df.drop(columns=["employee_id"])

# ---- Predict ----
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

# ---- Save Submission ----
submission = pd.DataFrame({
    "employee_id": emp_id,
    "is_promoted": preds,
    "promotion_probability": probs
})
submission.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Predictions saved to {OUTPUT_PATH}")
