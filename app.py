import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load saved models
marks_model = tf.keras.models.load_model("marks_model.keras")
grade_model = tf.keras.models.load_model("grade_model.keras")
risk_model = tf.keras.models.load_model("risk_model.keras")

# Load scaler and encoders
scaler = joblib.load("scaler.pkl")
grade_encoder = joblib.load("grade_encoder.pkl")
risk_encoder = joblib.load("risk_encoder.pkl")
parent_encoder = joblib.load("parent_support_encoder.pkl")
tuition_encoder = joblib.load("tuition_encoder.pkl")

# Streamlit UI
st.set_page_config(page_title="Student Performance Analyzer", layout="centered")

st.title("🎓 Student Performance Analyzer (Deep Learning)")
st.write("Enter student details below to predict **Marks, Grade, and Risk Level**.")

st.divider()

# Input fields
attendance = st.slider("📌 Attendance (%)", 0, 100, 75)
study_hours = st.slider("📚 Study Hours per Day", 0.0, 10.0, 4.0)
previous_marks = st.slider("📝 Previous Marks", 0, 100, 70)
assignment_score = st.slider("📑 Assignment Score", 0, 100, 75)
internal_marks = st.slider("🏫 Internal Marks (out of 50)", 0, 50, 30)

sleep_hours = st.slider("😴 Sleep Hours per Day", 0.0, 12.0, 7.0)
internet_hours = st.slider("📱 Internet Hours per Day", 0.0, 12.0, 4.0)
stress_level = st.slider("😟 Stress Level (1-10)", 1, 10, 5)

parent_support = st.selectbox("👨‍👩‍👧 Parent Support", ["Yes", "No"])
tuition = st.selectbox("📖 Tuition/Coaching", ["Yes", "No"])

st.divider()

# Prediction Button
if st.button("🔍 Predict Performance"):

    # Encode categorical inputs
    parent_support_encoded = parent_encoder.transform([parent_support])[0]
    tuition_encoded = tuition_encoder.transform([tuition])[0]

    # Create input array
    input_data = np.array([[attendance, study_hours, previous_marks,
                            assignment_score, internal_marks, sleep_hours,
                            internet_hours, stress_level,
                            parent_support_encoded, tuition_encoded]])

    # Scale input
    input_scaled = scaler.transform(input_data)
    
    predicted_marks = marks_model.predict(input_scaled)[0][0]

    # Fix: keep marks within 0 to 100
    predicted_marks = float(np.clip(predicted_marks, 0, 100))

    predicted_marks = round(predicted_marks, 2)


    # Predict grade
    grade_probs = grade_model.predict(input_scaled)
    grade_index = np.argmax(grade_probs)
    predicted_grade = grade_encoder.inverse_transform([grade_index])[0]

    # Predict risk
    risk_probs = risk_model.predict(input_scaled)
    risk_index = np.argmax(risk_probs)
    predicted_risk = risk_encoder.inverse_transform([risk_index])[0]

    # Display results
    st.subheader("Prediction Results")

    st.success(f"✅ Predicted Final Marks: **{predicted_marks} / 100**")
    st.info(f"🎓 Predicted Grade: **{predicted_grade}**")

    if predicted_risk == "Low":
        st.success(f"🟢 Risk Level: **{predicted_risk}**")
    elif predicted_risk == "Medium":
        st.warning(f"🟠 Risk Level: **{predicted_risk}**")
    else:
        st.error(f"🔴 Risk Level: **{predicted_risk}**")

    st.divider()

    # Recommendations Section
    st.subheader("💡 Suggestions to Improve Performance")

    suggestions = []

    if attendance < 75:
        suggestions.append("📌 Improve attendance to at least 75% for better results.")

    if study_hours < 3:
        suggestions.append("📌 Increase study hours to minimum 3 hours/day.")

    if assignment_score < 60:
        suggestions.append("📌 Focus on completing assignments properly.")

    if internal_marks < 25:
        suggestions.append("📌 Try to improve internal test performance.")

    if sleep_hours < 6:
        suggestions.append("📌 Sleep at least 6–8 hours for better concentration.")

    if internet_hours > 6:
        suggestions.append("📌 Reduce internet usage and avoid distractions.")

    if stress_level > 7:
        suggestions.append("📌 Stress is high. Practice meditation, exercise, or relaxation.")

    if len(suggestions) == 0:
        suggestions.append("✅ Great! Keep continuing your current routine.")

    for s in suggestions:
        st.write(s)

    st.divider()
    st.caption("Built using TensorFlow + Streamlit (Deep Learning Project)")









