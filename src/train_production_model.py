import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DATA_PATH = BASE_DIR / "data" / "processed" / "model_data.pkl"
MODEL_PATH = BASE_DIR / "data" / "processed" / "random_forest_model.pkl"


print("Loading prepared model data...")

X_train, X_test, y_train, y_test = joblib.load(
    MODEL_DATA_PATH
)

print(f"Training samples: {len(X_train)}")
print(f"Features: {X_train.shape[1]}")


print("Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


print("Saving model...")

joblib.dump(model, MODEL_PATH)


print()
print("Production model created successfully.")
print(f"Model path: {MODEL_PATH}")
print(f"Model type: {type(model).__name__}")
print(f"Number of trees: {model.n_estimators}")