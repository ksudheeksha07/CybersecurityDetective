from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


base_dir = Path(__file__).resolve().parent.parent

data_dir = (
    Path.home()
    / "Downloads"
    / "MachineLearningCSV"
    / "MachineLearningCVE"
    / "selected"
)

monday_file = data_dir / "Monday-WorkingHours.pcap_ISCX.csv"
friday_file = data_dir / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"


print("Loading Monday benign traffic...")
monday = pd.read_csv(monday_file)
monday.columns = monday.columns.str.strip()

print("Monday shape:", monday.shape)


print("\nLoading Friday DDoS traffic...")
friday = pd.read_csv(friday_file)
friday.columns = friday.columns.str.strip()

print("Friday shape:", friday.shape)


# ---------------------------------------------------------
# CLEAN DATA
# ---------------------------------------------------------

def clean_dataset(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.drop_duplicates()

    return df


monday = clean_dataset(monday)
friday = clean_dataset(friday)

print("\nCleaned Monday shape:", monday.shape)
print("Cleaned Friday shape:", friday.shape)


# ---------------------------------------------------------
# REMOVE CONSTANT FEATURES USING TRAINING DATA ONLY
# ---------------------------------------------------------

friday_features = friday.drop(columns=["Label"])

constant_features = [
    column
    for column in friday_features.columns
    if friday_features[column].nunique() <= 1
]

print("\nConstant features removed:", len(constant_features))


monday = monday.drop(
    columns=constant_features,
    errors="ignore"
)

friday = friday.drop(
    columns=constant_features,
    errors="ignore"
)


# ---------------------------------------------------------
# NUMERICAL FEATURES
# ---------------------------------------------------------

X_monday = monday.drop(columns=["Label"])
X_friday = friday.drop(columns=["Label"])

X_monday = X_monday.select_dtypes(include=np.number)
X_friday = X_friday.select_dtypes(include=np.number)


# Make sure both datasets use identical feature columns
common_features = [
    column
    for column in X_friday.columns
    if column in X_monday.columns
]

X_monday = X_monday[common_features]
X_friday = X_friday[common_features]


print("Features used:", len(common_features))


# ---------------------------------------------------------
# LABELS
# ---------------------------------------------------------

y_monday = monday["Label"]

y_friday = friday["Label"]


# ---------------------------------------------------------
# TRAINING DATA
# ---------------------------------------------------------

# Use Friday traffic for training.
# Preserve chronological order.

split_index = int(len(friday) * 0.70)

X_train_raw = X_friday.iloc[:split_index]
y_train = y_friday.iloc[:split_index]


# ---------------------------------------------------------
# FRIDAY UNSEEN TEMPORAL TEST
# ---------------------------------------------------------

X_friday_test_raw = X_friday.iloc[split_index:]
y_friday_test = y_friday.iloc[split_index:]


# ---------------------------------------------------------
# FIT PREPROCESSING ONLY ON TRAINING DATA
# ---------------------------------------------------------

imputer = SimpleImputer(
    missing_values=-1,
    strategy="median"
)

X_train = imputer.fit_transform(X_train_raw)

X_friday_test = imputer.transform(
    X_friday_test_raw
)

X_monday_test = imputer.transform(
    X_monday
)


# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)


# ---------------------------------------------------------
# FRIDAY TEMPORAL EVALUATION
# ---------------------------------------------------------

print("\nEvaluating on unseen Friday time period...")

friday_pred = model.predict(X_friday_test)

friday_accuracy = accuracy_score(
    y_friday_test,
    friday_pred
)

friday_precision = precision_score(
    y_friday_test,
    friday_pred,
    pos_label="DDoS"
)

friday_recall = recall_score(
    y_friday_test,
    friday_pred,
    pos_label="DDoS"
)

friday_f1 = f1_score(
    y_friday_test,
    friday_pred,
    pos_label="DDoS"
)

tn, fp, fn, tp = confusion_matrix(
    y_friday_test,
    friday_pred,
    labels=["BENIGN", "DDoS"]
).ravel()


# ---------------------------------------------------------
# MONDAY BENIGN-DAY EVALUATION
# ---------------------------------------------------------

print("\nEvaluating on completely separate Monday benign traffic...")

monday_pred = model.predict(X_monday_test)

monday_false_positive_rate = np.mean(
    monday_pred == "DDoS"
)

monday_false_positives = int(
    np.sum(monday_pred == "DDoS")
)


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CROSS-DAY EVALUATION RESULTS")
print("=" * 70)

print("\nTraining samples:", len(X_train))
print("Friday temporal test samples:", len(X_friday_test))
print("Monday unseen benign samples:", len(X_monday_test))
print("Features:", len(common_features))


print("\n--- Friday temporal test ---")

print("Accuracy:", round(friday_accuracy, 4))
print("Precision:", round(friday_precision, 4))
print("Recall:", round(friday_recall, 4))
print("F1:", round(friday_f1, 4))

print("False Positives:", fp)
print("False Negatives:", fn)


print("\n--- Monday unseen benign test ---")

print(
    "False positives:",
    monday_false_positives
)

print(
    "False positive rate:",
    round(monday_false_positive_rate, 6)
)


print("\n" + "=" * 70)
print("CROSS-DAY EVALUATION COMPLETE")
print("=" * 70)


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

output_file = (
    base_dir
    / "experiments"
    / "cross_day_evaluation.txt"
)

with open(output_file, "w") as f:

    f.write("CROSS-DAY EVALUATION\n")
    f.write("=" * 70 + "\n\n")

    f.write(
        f"Training samples: {len(X_train)}\n"
    )

    f.write(
        f"Friday temporal test samples: {len(X_friday_test)}\n"
    )

    f.write(
        f"Monday unseen benign samples: {len(X_monday_test)}\n"
    )

    f.write(
        f"Features: {len(common_features)}\n\n"
    )

    f.write("Friday temporal test\n")
    f.write("-" * 30 + "\n")

    f.write(
        f"Accuracy: {friday_accuracy:.4f}\n"
    )

    f.write(
        f"Precision: {friday_precision:.4f}\n"
    )

    f.write(
        f"Recall: {friday_recall:.4f}\n"
    )

    f.write(
        f"F1: {friday_f1:.4f}\n"
    )

    f.write(
        f"False Positives: {fp}\n"
    )

    f.write(
        f"False Negatives: {fn}\n\n"
    )

    f.write("Monday unseen benign test\n")
    f.write("-" * 30 + "\n")

    f.write(
        f"False positives: {monday_false_positives}\n"
    )

    f.write(
        f"False positive rate: "
        f"{monday_false_positive_rate:.6f}\n"
    )


print("\nResults saved to:")
print(output_file)