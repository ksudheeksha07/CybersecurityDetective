from pathlib import Path

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split


base_dir = Path(__file__).resolve().parent.parent

data_file = base_dir / "data" / "processed" / "train_final.csv"
pipeline_file = base_dir / "data" / "processed" / "preprocessing_pipeline.pkl"

print("Loading final dataset...")

df = pd.read_csv(data_file)
df.columns = df.columns.str.strip()

X = df.drop(columns=["Label"])
y = df["Label"]

print("Total samples:", len(df))
print("Features:", X.shape[1])

print("\nApplying preprocessing pipeline...")

preprocessor = joblib.load(pipeline_file)

X_processed = preprocessor.transform(X)

print("Processed shape:", X_processed.shape)

print("\nCreating stratified train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("MODEL DATA PREPARATION")
print("=" * 60)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Features:", X_train.shape[1])

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

output_dir = base_dir / "data" / "processed"

joblib.dump(
    (X_train, X_test, y_train, y_test),
    output_dir / "model_data.pkl"
)

print("\nModel data saved to:")
print(output_dir / "model_data.pkl")

print("\n✓ MODEL DATA PREPARATION COMPLETE")