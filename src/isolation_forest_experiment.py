from pathlib import Path
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score


# Project paths
base_dir = Path(__file__).resolve().parent.parent

train_file = base_dir / "data" / "processed" / "train.csv"
test_file = base_dir / "data" / "processed" / "test.csv"


# Load data
print("Loading training and testing data...")

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# Clean column names
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()


# Separate labels
y_test = test_df["Label"]

X_train = train_df.drop(columns=["Label"])
X_test = test_df.drop(columns=["Label"])


# Keep numeric features
X_train = X_train.select_dtypes(include=["number"])
X_test = X_test[X_train.columns]


# Convert actual labels:
# BENIGN = 0
# DDoS = 1
actual_labels = y_test.map({
    "BENIGN": 0,
    "DDoS": 1
})


# Contamination values to test
contamination_values = [0.10, 0.20, 0.30, 0.40, 0.50]


results = []


for contamination in contamination_values:

    print("\n" + "=" * 60)
    print(f"Testing contamination = {contamination}")

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train)

    predictions = model.predict(X_test)

    # Isolation Forest:
    #  1  = normal
    # -1  = anomaly
    #
    # Convert:
    # normal  -> 0
    # anomaly -> 1

    predicted_labels = (predictions == -1).astype(int)

    precision = precision_score(
        actual_labels,
        predicted_labels,
        zero_division=0
    )

    recall = recall_score(
        actual_labels,
        predicted_labels,
        zero_division=0
    )

    f1 = f1_score(
        actual_labels,
        predicted_labels,
        zero_division=0
    )

    anomalies = (predictions == -1).sum()

    print("Anomalies detected:", anomalies)
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))

    results.append({
        "Contamination": contamination,
        "Anomalies": anomalies,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })


# Final comparison
results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("ISOLATION FOREST CONTAMINATION EXPERIMENT")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1": "{:.4f}".format
        }
    )
)


# Save results
output_file = base_dir / "experiments" / "isolation_forest_contamination.csv"

results_df.to_csv(output_file, index=False)

print("\nResults saved to:")
print(output_file)