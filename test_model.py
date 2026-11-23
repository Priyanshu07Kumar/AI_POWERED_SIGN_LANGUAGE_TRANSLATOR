import pandas as pd
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent

print("Loading trained model...")

model = joblib.load(BASE_DIR / "isl_model.pkl")

print("Model loaded successfully!")

# Load unseen testing dataset
data = pd.read_csv(BASE_DIR / "isl_test_dataset.csv")

print("Testing samples:", len(data))

# Get exact feature names from trained model
features = list(model.feature_names_in_)

# Check dataset format
missing = set(features) - set(data.columns)

if missing:
    raise ValueError(f"Missing features: {missing}")

if "gesture" not in data.columns:
    raise ValueError("Missing gesture column!")

# Separate features and actual labels
X_test = data[features]
y_test = data["gesture"]

# Predict gestures
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\n========== MODEL TEST RESULTS ==========")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, predictions))

print("\nTesting completed!")
