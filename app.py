import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("diabetes_model.pkl")

# Feature list
feature_names = [
    'high_blood_pressure', 'high_cholesterol', 'cholesterol_check', 'BMI',
    'is_smoker', 'had_stroke', 'heart_disease', 'physically_active',
    'eats_fruits', 'eats_vegetables', 'heavy_alcohol_consumption',
    'has_healthcare', 'no_doctor_due_to_cost', 'general_health',
    'mentally_healthy', 'physically_healthy', 'difficulty_walking',
    'gender', 'age', 'education_level', 'income_level',
    'health_score', 'physically_inactive', 'lifestyle_risk_score'
]

# UI
st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")
st.title("🩺 Diabetes Risk Predictor")
st.markdown("Enter your health details below to check your diabetes risk.")

# Inputs
def get_user_input():
    inputs = {}

    for name in feature_names:
        display_name = name.replace('_', ' ').capitalize()

        if name in ['BMI', 'age', 'health_score', 'mentally_healthy', 'physically_healthy', 'lifestyle_risk_score']:
            inputs[name] = st.number_input(display_name, min_value=0.0)

        elif name == 'general_health':
            options = {
                "Excellent": 1,
                "Very Good": 2,
                "Good": 3,
                "Fair": 4,
                "Poor": 5
            }
            choice = st.selectbox("General Health", list(options.keys()))
            inputs[name] = options[choice]

        elif name == 'education_level':
            options = {
                "Never Attended School": 1,
                "Elementary (Grades 1–8)": 2,
                "Some High School": 3,
                "High School Graduate": 4,
                "College or Above": 5
            }
            choice = st.selectbox("Education Level", list(options.keys()))
            inputs[name] = options[choice]

        elif name == 'income_level':
            options = {
                "Less than $10,000": 1,
                "$10,000–$19,999": 2,
                "$20,000–$34,999": 3,
                "$35,000–$49,999": 4,
                "$50,000 or more": 5
            }
            choice = st.selectbox("Income Level", list(options.keys()))
            inputs[name] = options[choice]

        elif name == 'gender':
            gender = st.selectbox("Gender", ["Female", "Male"])
            inputs[name] = 1 if gender == "Male" else 0

        elif name in ['high_blood_pressure', 'high_cholesterol', 'cholesterol_check', 'is_smoker',
                      'had_stroke', 'heart_disease', 'physically_active', 'eats_fruits',
                      'eats_vegetables', 'heavy_alcohol_consumption', 'has_healthcare',
                      'no_doctor_due_to_cost', 'difficulty_walking', 'physically_inactive']:
            yn = st.selectbox(display_name, ["No", "Yes"])
            inputs[name] = 1 if yn == "Yes" else 0

        else:
            # Fallback to number input for any unexpected feature
            inputs[name] = st.number_input(display_name, min_value=0.0)

    return np.array([list(inputs.values())])

# Prediction
user_input = get_user_input()
if st.button("Predict"):
    prediction = model.predict(user_input)
    probability = model.predict_proba(user_input)[0][1]

    st.subheader("🧾 Prediction Result")
    if prediction[0] == 1:
        st.error(f"⚠️ High risk of diabetes with probability {probability:.2f}")
    else:
        st.success(f"✅ Low risk of diabetes with probability {probability:.2f}")

    st.markdown("🔍 This prediction is based on your current lifestyle and health indicators.")
