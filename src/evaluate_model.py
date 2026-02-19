import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_preprocessing import load_data

# load model
model = joblib.load("../models/model.pkl")

# load data
df = load_data()

X = df.drop("Health_Risk_Level", axis=1)
y = df["Health_Risk_Level"]

pred = model.predict(X)

print("Accuracy:", accuracy_score(y, pred))
print("\nClassification Report:\n", classification_report(y, pred))
print("\nConfusion Matrix:\n", confusion_matrix(y, pred))
