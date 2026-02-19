import streamlit as st
import pandas as pd
import joblib
import os

# Load model
model = joblib.load("models/model.pkl")

st.title("Student Health Risk Predictor")

# Inputs
age = st.slider("Age", 18, 30, 22)
heart = st.number_input("Heart Rate", 50, 120, 72)
sys = st.number_input("Systolic BP", 90, 160, 120)
dia = st.number_input("Diastolic BP", 60, 120, 80)
bio = st.slider("Stress Biosensor", 0.0, 10.0, 5.0)
self_r = st.slider("Stress Self Report", 0.0, 10.0, 5.0)
study = st.slider("Study Hours", 0.0, 12.0, 5.0)
project = st.slider("Project Hours", 0.0, 12.0, 3.0)
family = st.slider("Family Members", 1, 10, 4)

gender = st.selectbox("Gender", ["Male","Female"])
activity = st.selectbox("Physical Activity", ["Low","Moderate","High"])
sleep = st.selectbox("Sleep Quality", ["Poor","Moderate","Good"])
mood = st.selectbox("Mood", ["Happy","Sad","Stressed","Neutral"])

# Predict button
if st.button("Predict"):

    sample = pd.DataFrame({
        "Age":[age],
        "Heart_Rate":[heart],
        "Blood_Pressure_Systolic":[sys],
        "Blood_Pressure_Diastolic":[dia],
        "Stress_Level_Biosensor":[bio],
        "Stress_Level_Self_Report":[self_r],
        "Study_Hours":[study],
        "Project_Hours":[project],
        "Family_members":[family],
        "Gender":[gender],
        "Physical_Activity":[activity],
        "Sleep_Quality":[sleep],
        "Mood":[mood]
    })

    pred = model.predict(sample)[0]

    st.success(f"Predicted Health Risk Level: {pred}")
