from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


train_file = Path("data/processed/train.csv")

train_data = pd.read_csv(train_file)

X_train = train_data.drop(columns=[" Label"])
y_train = train_data[" Label"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("=" * 70)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 features:")
print(importance.head(15).to_string(index=False))
