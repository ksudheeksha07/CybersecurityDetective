from pathlib import Path

import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


base_dir = Path(__file__).resolve().parent.parent

model_data_file = base_dir / "data" / "processed" / "model_data.pkl"

print("Loading prepared model data...")

X_train, X_test, y_train, y_test = joblib.load(model_data_file)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Features:", X_train.shape[1])

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

print("\nMaking predictions...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 60)

print("\nAccuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

output_file = (
    base_dir
    / "experiments"
    / "logistic_regression_results.txt"
)

with open(output_file, "w") as f:
    f.write("LOGISTIC REGRESSION RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(classification_report(y_test, y_pred))

print("\nResults saved to:")
print(output_file)

print("\n✓ LOGISTIC REGRESSION EXPERIMENT COMPLETE")