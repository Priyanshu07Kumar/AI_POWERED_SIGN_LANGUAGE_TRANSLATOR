import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Loading ISL dataset...")

# 1. Load dataset
data = pd.read_csv("isl_dataset.csv")

print("Dataset loaded!")
print("Total samples:", len(data))

# 2. Separate features and labels
X = data.drop(columns=["gesture"])
y = data["gesture"]

# 3. Check missing values
if X.isnull().sum().sum() > 0:
    print("Missing values found. Filling with zeros...")
    X = X.fillna(0)

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 5. Create Random Forest model
print("\nTraining AI model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# 6. Train
model.fit(X_train, y_train)

print("Training completed!")

# 7. Test model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n========== MODEL RESULTS ==========")

print("Model Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# 8. Save model
joblib.dump(model, "isl_model.pkl")

print("\nModel saved successfully!")
print("File: isl_model.pkl")