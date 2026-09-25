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

train_file = (
    base_dir
    / "data"
    / "processed"
    / "train_final.csv"
)

test_file = (
    base_dir
    / "data"
    / "processed"
    / "test.csv"
)


print("Loading training data...")
train = pd.read_csv(train_file)

print("Loading testing data...")
test = pd.read_csv(test_file)


train.columns = train.columns.str.strip()
test.columns = test.columns.str.strip()


# ---------------------------------------------------------
# PREPARE FEATURES
# ---------------------------------------------------------

X_train = train.drop(columns=["Label"])
y_train = train["Label"]

X_test = test.drop(columns=["Label"])
y_test = test["Label"]


# Keep numerical features only
X_train = X_train.select_dtypes(include=np.number)
X_test = X_test.select_dtypes(include=np.number)


# Use the same feature columns
common_features = [
    column
    for column in X_train.columns
    if column in X_test.columns
]

X_train = X_train[common_features]
X_test = X_test[common_features]


print("\nTotal features:", len(common_features))


# ---------------------------------------------------------
# HANDLE SENTINEL VALUES
# ---------------------------------------------------------

imputer = SimpleImputer(
    missing_values=-1,
    strategy="median"
)

X_train_processed = imputer.fit_transform(X_train)
X_test_processed = imputer.transform(X_test)


# ---------------------------------------------------------
# BASELINE FEATURE IMPORTANCE
# ---------------------------------------------------------

print("\nTraining baseline Random Forest...")

baseline_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

baseline_model.fit(
    X_train_processed,
    y_train
)


importance = pd.Series(
    baseline_model.feature_importances_,
    index=common_features
)

importance = importance.sort_values(
    ascending=False
)


# ---------------------------------------------------------
# FEATURE COUNTS TO TEST
# ---------------------------------------------------------

feature_counts = [
    68,
    50,
    30,
    20,
    10
]


results = []


# ---------------------------------------------------------
# ABLATION EXPERIMENT
# ---------------------------------------------------------

for count in feature_counts:

    selected_features = (
        importance
        .head(count)
        .index
        .tolist()
    )

    feature_indices = [
        common_features.index(feature)
        for feature in selected_features
    ]

    X_train_subset = X_train_processed[
        :, feature_indices
    ]

    X_test_subset = X_test_processed[
        :, feature_indices
    ]


    print(
        f"\nTraining Random Forest "
        f"with top {count} features..."
    )


    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_subset,
        y_train
    )


    y_pred = model.predict(
        X_test_subset
    )


    accuracy = accuracy_score(
        y_test,
        y_pred
    )

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


    results.append(
        {
            "features": count,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

results_df = pd.DataFrame(results)


print("\n" + "=" * 70)
print("FEATURE ABLATION RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

output_file = (
    base_dir
    / "experiments"
    / "feature_ablation.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


print("\nResults saved to:")
print(output_file)

print("\n✓ FEATURE ABLATION COMPLETE")