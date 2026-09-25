from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


train_file = Path("data/processed/train.csv")
test_file = Path("data/processed/test.csv")

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

X_train = train_data.drop(columns=[" Label"])
y_train = train_data[" Label"]

X_test = test_data.drop(columns=[" Label"])
y_test = test_data[" Label"]

# Scale features and train Logistic Regression
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

print("=" * 70)
print("LOGISTIC REGRESSION BASELINE")
print("=" * 70)

print("\nTraining model...")
model.fit(X_train, y_train)

print("Training complete.")

print("\nTest set evaluation:")
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))