import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load normalized dataset
df = pd.read_csv("isl_normalized_dataset.csv")

X = df.drop(columns=["gesture"])
y = df["gesture"]

print("Dataset loaded:", len(df))
print("Gestures:", sorted(y.unique()))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

print("\nTraining AI model...")
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nTraining completed!")
print("Test accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save as a separate model
joblib.dump(model, "isl_normalized_model.pkl")

print("\nNew model saved: isl_normalized_model.pkl")