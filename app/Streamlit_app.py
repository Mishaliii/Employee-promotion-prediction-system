import streamlit as st
import pandas as pd
import joblib

# ---- Load trained model ----
MODEL_PATH = "../models/logreg_promotion_pipeline.pkl"
model = joblib.load(MODEL_PATH)

# ---- Page Title ----
st.title("🏆 Employee Promotion Prediction System")
st.markdown(
    "This demo uses a machine-learning model trained on historical HR data "
    "to estimate the likelihood of an employee being promoted in the next cycle."
)
st.info("Fill in the details below and click **Predict Promotion** to see the result.")

# ---- Input fields ----
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
awards_won = st.selectbox("Awards Won?", [0, 1])
avg_training_score = st.number_input("Average Training Score", min_value=0, max_value=100, value=60)

# ---- Predict Button ----
if st.button("Predict Promotion"):
    # Build dataframe for input
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

    # Predict
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    # Display result
    if pred == 1:
        st.success(f"✅ This employee is **likely to be promoted** (probability: {prob:.2%})")
    else:
        st.error(f"❌ This employee is **unlikely to be promoted** (probability: {prob:.2%})")

    st.progress(int(prob * 100))


