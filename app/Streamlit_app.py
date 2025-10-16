import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---- Load model & results ----
MODEL_PATH = "../models/best_model.pkl"
COMPARISON_PATH = "../models/model_comparison.csv"

model = joblib.load(MODEL_PATH)
comparison_df = pd.read_csv(COMPARISON_PATH)

# ---- Title & Description ----
st.title("🏆 Employee Promotion Prediction System")
st.markdown(
    "This app compares five machine learning algorithms — "
    "**KNN, Logistic Regression, Naive Bayes, Decision Tree, and SVM** — "
    "and automatically uses the model with the highest accuracy "
    "for predicting whether an employee is likely to be promoted."
)

# ---- Display Comparison Table ----
st.subheader("📊 Model Accuracy Comparison")
st.dataframe(comparison_df.style.format({
    "Accuracy": "{:.2%}", "Balanced Accuracy": "{:.2%}", "ROC-AUC": "{:.2f}"
}))

# ---- Accuracy Bar Chart ----
st.subheader("🔍 Accuracy Visualization")
fig, ax = plt.subplots()
ax.bar(comparison_df["Model"], comparison_df["Accuracy"], color="skyblue")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
ax.set_title("Model Accuracy Comparison")
st.pyplot(fig)

st.divider()

# ---- Input Section ----
st.header("📝 Employee Details")

department = st.selectbox("Department", [
    "Sales & Marketing", "Operations", "Technology", "Procurement",
    "Finance", "HR", "Legal"
])
region = st.selectbox("Region", [
    "region_1","region_2","region_3","region_4","region_7","region_13","region_19",
    "region_22","region_23","region_26","region_29","region_31","region_33","region_34","region_46"
])
education = st.selectbox("Education", ["Bachelor's", "Master's & above", "Below Secondary"])
gender = st.selectbox("Gender", ["m", "f"])
recruitment_channel = st.selectbox("Recruitment Channel", ["sourcing", "other", "referred"])
no_of_trainings = st.number_input("Number of Trainings", min_value=0, max_value=10, value=1)
age = st.number_input("Age", min_value=18, max_value=60, value=30)
previous_year_rating = st.number_input("Previous Year Rating (1-5)", min_value=1.0, max_value=5.0, value=3.0)
length_of_service = st.number_input("Length of Service (years)", min_value=0, max_value=40, value=5, step=1)
awards_won = st.selectbox("Awards Won?", [0, 1, 2, 3, 4, 5])
avg_training_score = st.number_input("Average Training Score", min_value=0, max_value=100, value=60)

# ---- Prediction ----
if st.button("Predict Promotion"):
    input_df = pd.DataFrame([{
        "department": department,
        "region": region,
        "education": education,
        "gender": gender,
        "recruitment_channel": recruitment_channel,
        "no_of_trainings": no_of_trainings,
        "age": age,
        "previous_year_rating": previous_year_rating,
        "length_of_service": length_of_service,
        "awards_won?": awards_won,
        "avg_training_score": avg_training_score
    }])

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else 0

    st.subheader("🎯 Prediction Result")
    if pred == 1:
        st.success(f"✅ This employee is **likely to be promoted** (probability: {prob:.2%})")
    else:
        st.error(f"❌ This employee is **unlikely to be promoted** (probability: {prob:.2%})")

    st.progress(int(prob * 100))


