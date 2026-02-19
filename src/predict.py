import joblib
import pandas as pd
import os

# get project root path dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# model path
model_path = os.path.join(BASE_DIR, "models", "model.pkl")

# load model
model = joblib.load(model_path)

# sample input
sample = pd.DataFrame({
    "Age":[22],
    "Heart_Rate":[72],
    "Blood_Pressure_Systolic":[120],
    "Blood_Pressure_Diastolic":[80],
    "Stress_Level_Biosensor":[4.5],
    "Stress_Level_Self_Report":[6],
    "Study_Hours":[5],
    "Project_Hours":[3],
    "Family_members":[4],
    "Gender":["Male"],
    "Physical_Activity":["Moderate"],
    "Sleep_Quality":["Good"],
    "Mood":["Happy"]
})

# prediction
pred = model.predict(sample)

print("Predicted Health Risk Level:", pred[0])
