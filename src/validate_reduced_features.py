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


print("Loading Monday traffic...")
monday = pd.read_csv(monday_file)
monday.columns = monday.columns.str.strip()

print("Loading Friday traffic...")
friday = pd.read_csv(friday_file)
friday.columns = friday.columns.str.strip()


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
# FEATURE SET
# ---------------------------------------------------------

friday_features = friday.drop(columns=["Label"])

constant_features = [
    column
    for column in friday_features.columns
    if friday_features[column].nunique() <= 1
]

monday = monday.drop(
    columns=constant_features,
    errors="ignore"
)

friday = friday.drop(
    columns=constant_features,
    errors="ignore"
)


X_monday = monday.drop(columns=["Label"])
X_friday = friday.drop(columns=["Label"])

X_monday = X_monday.select_dtypes(include=np.number)
X_friday = X_friday.select_dtypes(include=np.number)


common_features = [
    column
    for column in X_friday.columns
    if column in X_monday.columns
]

X_monday = X_monday[common_features]
X_friday = X_friday[common_features]

y_monday = monday["Label"]
y_friday = friday["Label"]


print("Total available features:", len(common_features))


# ---------------------------------------------------------
# TEMPORAL TRAIN / TEST SPLIT
# ---------------------------------------------------------

split_index = int(len(friday) * 0.70)

X_train_raw = X_friday.iloc[:split_index]
y_train = y_friday.iloc[:split_index]

X_friday_test_raw = X_friday.iloc[split_index:]
y_friday_test = y_friday.iloc[split_index:]


# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------

imputer = SimpleImputer(
    missing_values=-1,
    strategy="median"
)

X_train_processed = imputer.fit_transform(
    X_train_raw
)

X_friday_test_processed = imputer.transform(
    X_friday_test_raw
)

X_monday_processed = imputer.transform(
    X_monday
)


# ---------------------------------------------------------
# FEATURE IMPORTANCE FROM TRAINING DATA ONLY
# ---------------------------------------------------------

print("\nTraining baseline model for feature ranking...")

ranking_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

ranking_model.fit(
    X_train_processed,
    y_train
)


importance = pd.Series(
    ranking_model.feature_importances_,
    index=common_features
)

importance = importance.sort_values(
    ascending=False
)


top_50_features = (
    importance
    .head(50)
    .index
    .tolist()
)


feature_indices = [
    common_features.index(feature)
    for feature in top_50_features
]


X_train_50 = X_train_processed[
    :, feature_indices
]

X_friday_test_50 = X_friday_test_processed[
    :, feature_indices
]

X_monday_50 = X_monday_processed[
    :, feature_indices
]


print("\nSelected top 50 features:")
for i, feature in enumerate(top_50_features, start=1):
    print(f"{i:2}. {feature}")


# ---------------------------------------------------------
# TRAIN REDUCED MODEL
# ---------------------------------------------------------

print("\nTraining Random Forest with top 50 features...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train_50,
    y_train
)


# ---------------------------------------------------------
# FRIDAY TEMPORAL TEST
# ---------------------------------------------------------

print("\nEvaluating Friday temporal test...")

friday_pred = model.predict(
    X_friday_test_50
)

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


# ---------------------------------------------------------
# MONDAY BENIGN TEST
# ---------------------------------------------------------

print("\nEvaluating Monday unseen benign traffic...")

monday_pred = model.predict(
    X_monday_50
)

monday_false_positives = int(
    np.sum(monday_pred == "DDoS")
)

monday_fpr = (
    monday_false_positives
    / len(monday_pred)
)


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TOP-50 FEATURE VALIDATION")
print("=" * 70)

print("\nFeatures used:", len(top_50_features))

print("\n--- Friday temporal test ---")
print("Accuracy:", round(friday_accuracy, 4))
print("Precision:", round(friday_precision, 4))
print("Recall:", round(friday_recall, 4))
print("F1:", round(friday_f1, 4))

print("\n--- Monday unseen benign test ---")
print("False positives:", monday_false_positives)
print("False positive rate:", round(monday_fpr, 6))


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

output_file = (
    base_dir
    / "experiments"
    / "top50_temporal_validation.txt"
)

with open(output_file, "w") as f:

    f.write("TOP-50 FEATURE TEMPORAL VALIDATION\n")
    f.write("=" * 70 + "\n\n")

    f.write(
        f"Features used: {len(top_50_features)}\n"
    )

    f.write("\nFriday temporal test\n")
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

    f.write("\nMonday unseen benign test\n")
    f.write("-" * 30 + "\n")

    f.write(
        f"False positives: "
        f"{monday_false_positives}\n"
    )

    f.write(
        f"False positive rate: "
        f"{monday_fpr:.6f}\n"
    )

print("\nResults saved to:")
print(output_file)

print("\n✓ TOP-50 FEATURE VALIDATION COMPLETE")