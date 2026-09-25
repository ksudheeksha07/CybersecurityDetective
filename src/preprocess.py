from pathlib import Path

import pandas as pd
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


base_dir = Path(__file__).resolve().parent.parent

train_file = base_dir / "data" / "processed" / "train_final.csv"
output_dir = base_dir / "data" / "processed"

pipeline_file = output_dir / "preprocessing_pipeline.pkl"


print("Loading final training dataset...")

df = pd.read_csv(train_file)
df.columns = df.columns.str.strip()

X = df.drop(columns=["Label"])
y = df["Label"]

print("\nFeatures:", X.shape[1])
print("Training samples:", X.shape[0])

print("\nCreating preprocessing pipeline...")

preprocessor = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                missing_values=-1,
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

X_processed = preprocessor.fit_transform(X)

print("\nPreprocessing completed!")

print("Processed feature shape:", X_processed.shape)

joblib.dump(preprocessor, pipeline_file)

print("\nPreprocessing pipeline saved to:")
print(pipeline_file)

print("\n✓ PREPROCESSING PIPELINE CREATED")