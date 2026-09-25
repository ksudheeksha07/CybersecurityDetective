from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


train_file = Path("data/processed/train.csv")
test_file = Path("data/processed/test.csv")

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

X_train = train_data.drop(columns=[" Label"])
y_train = train_data[" Label"]

X_test = test_data.drop(columns=[" Label"])
y_test = test_data[" Label"]

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

print("=" * 70)
print("RANDOM FOREST BASELINE")
print("=" * 70)

print("\nTraining model...")
model.fit(X_train, y_train)

print("Training complete.")

print("\nTest set evaluation:")
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))