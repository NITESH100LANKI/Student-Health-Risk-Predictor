from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

from data_preprocessing import load_data, get_preprocessor

# load data
df = load_data()

# split
X = df.drop("Health_Risk_Level", axis=1)
y = df["Health_Risk_Level"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# preprocessing
preprocessor = get_preprocessor(df)

# model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# train
model.fit(x_train, y_train)

# save
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(BASE_DIR, "models")

os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "model.pkl")

joblib.dump(model, model_path)


print("Model trained & saved successfully ✅")
