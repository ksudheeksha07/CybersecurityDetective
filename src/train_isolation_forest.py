from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer


# =========================================================
# Project paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "train.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "isolation_forest_benign_model.pkl"
)

IMPUTER_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "anomaly_benign_imputer.pkl"
)

REFERENCE_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "anomaly_benign_score_reference.pkl"
)


# =========================================================
# Load training data
# =========================================================

print("Loading training data...")

train_df = pd.read_csv(TRAIN_FILE)

# Remove accidental whitespace from dataset column names.
train_df.columns = train_df.columns.str.strip()

print("Training samples:", len(train_df))
print("Columns:", len(train_df.columns))


# =========================================================
# Verify labels
# =========================================================

if "Label" not in train_df.columns:
    raise ValueError("Label column not found.")

print("\nClass distribution:")
print(train_df["Label"].value_counts())


# =========================================================
# Keep BENIGN traffic only
# =========================================================

benign_df = train_df[
    train_df["Label"] == "BENIGN"
].copy()

if benign_df.empty:
    raise ValueError("No BENIGN samples found.")


print(
    "\nBENIGN samples used for anomaly training:",
    len(benign_df)
)


# =========================================================
# Separate features
# =========================================================

X_benign = benign_df.drop(
    columns=["Label"]
)

# Keep numeric features only.
X_benign = X_benign.select_dtypes(
    include=["number"]
)

print(
    "Numeric features:",
    X_benign.shape[1]
)


# =========================================================
# Handle invalid values
# =========================================================

# Convert infinite values to NaN.
X_benign = X_benign.replace(
    [np.inf, -np.inf],
    np.nan
)


# The dataset contains -1 sentinel values.
# Treat them as missing values for anomaly detection.
X_benign = X_benign.replace(
    -1,
    np.nan
)


# =========================================================
# Imputation
# =========================================================

print("\nFitting anomaly imputer...")

imputer = SimpleImputer(
    strategy="median"
)

X_benign_imputed = imputer.fit_transform(
    X_benign
)


# =========================================================
# Train BENIGN-only Isolation Forest
# =========================================================

print("\nTraining BENIGN-only Isolation Forest...")

model = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

model.fit(X_benign_imputed)


# =========================================================
# Calculate training decision-score distribution
# =========================================================

print(
    "\nCalculating BENIGN training score distribution..."
)

training_decision_scores = (
    model.decision_function(
        X_benign_imputed
    )
)


# =========================================================
# Score reference
# =========================================================

reference = {
    "p01": float(
        np.percentile(
            training_decision_scores,
            1
        )
    ),
    "p05": float(
        np.percentile(
            training_decision_scores,
            5
        )
    ),
    "p50": float(
        np.percentile(
            training_decision_scores,
            50
        )
    ),
    "p95": float(
        np.percentile(
            training_decision_scores,
            95
        )
    ),
    "p99": float(
        np.percentile(
            training_decision_scores,
            99
        )
    ),
}


# =========================================================
# Save artifacts
# =========================================================

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    imputer,
    IMPUTER_PATH
)

joblib.dump(
    reference,
    REFERENCE_PATH
)


# =========================================================
# Results
# =========================================================

print("\n" + "=" * 60)
print("BENIGN-ONLY ISOLATION FOREST")
print("=" * 60)

print(
    "Training samples:",
    len(X_benign)
)

print(
    "Features:",
    X_benign.shape[1]
)

print(
    "Model:",
    MODEL_PATH.name
)

print(
    "Imputer:",
    IMPUTER_PATH.name
)

print(
    "Reference:",
    REFERENCE_PATH.name
)

print("\nTraining decision-score reference:")

for key, value in reference.items():
    print(
        f"{key}: {value:.10f}"
    )

print(
    "\nBENIGN-only anomaly model trained successfully."
)