from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

base_dir = Path(__file__).resolve().parent.parent

data_file = (
    Path.home()
    / "Downloads"
    / "MachineLearningCSV"
   / "MachineLearningCVE"
/ "selected"
/ "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

print("Loading original Friday DDoS traffic...")

df = pd.read_csv(data_file)
df.columns = df.columns.str.strip()

print("Original shape:", df.shape)

# Replace infinite values
df = df.replace([np.inf, -np.inf], np.nan)

# Remove rows with missing values
df = df.dropna()

# Remove duplicate rows
df = df.drop_duplicates()

print("Cleaned shape:", df.shape)

# Remove constant features
feature_columns = df.drop(columns=["Label"]).columns

constant_features = [
    column
    for column in feature_columns
    if df[column].nunique() <= 1
]

print("\nConstant features removed:", len(constant_features))

df = df.drop(columns=constant_features)

# Separate features and labels
X = df.drop(columns=["Label"])
y = df["Label"]

# Keep numerical features only
X = X.select_dtypes(include=np.number)

print("Features used:", X.shape[1])

# Handle sentinel -1 values
imputer = SimpleImputer(
    missing_values=-1,
    strategy="median"
)

X_processed = imputer.fit_transform(X)

# Chronological split
split_index = int(len(df) * 0.70)

X_train = X_processed[:split_index]
X_test = X_processed[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\n" + "=" * 70)
print("TEMPORAL EVALUATION")
print("=" * 70)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining distribution:")
print(y_train.value_counts())

print("\nTesting distribution:")
print(y_test.value_counts())

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Making predictions...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    pos_label="DDoS"
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label="DDoS"
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label="DDoS"
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred,
    labels=["BENIGN", "DDoS"]
).ravel()

print("\n" + "=" * 70)
print("FINAL TEMPORAL RESULTS")
print("=" * 70)

print("\nAccuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1:", round(f1, 4))

print("\nFalse Positives:", fp)
print("False Negatives:", fn)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(
    pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual BENIGN", "Actual DDoS"],
        columns=["Predicted BENIGN", "Predicted DDoS"]
    )
)

output_file = (
    base_dir
    / "experiments"
    / "temporal_evaluation_final.txt"
)

with open(output_file, "w") as f:
    f.write("FINAL TEMPORAL EVALUATION\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Testing samples: {len(X_test)}\n")
    f.write(f"Features: {X_train.shape[1]}\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1: {f1:.4f}\n")
    f.write(f"False Positives: {fp}\n")
    f.write(f"False Negatives: {fn}\n\n")
    f.write(classification_report(y_test, y_pred))

print("\nResults saved to:")
print(output_file)

print("\n✓ TEMPORAL EVALUATION COMPLETE")