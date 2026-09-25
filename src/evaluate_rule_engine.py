from pathlib import Path
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# Project paths
base_dir = Path(__file__).resolve().parent.parent

results_file = (
    base_dir
    / "experiments"
    / "rule_engine_results.csv"
)


# Load rule-engine results
print("Loading rule-engine results...")

df = pd.read_csv(results_file)

print("Rows:", len(df))


# Convert labels
actual = df["Label"].map({
    "BENIGN": 0,
    "DDoS": 1
})

# Rule alert:
# False = 0
# True  = 1
predicted = df["Rule_Alert"].astype(int)


# Calculate metrics
precision = precision_score(
    actual,
    predicted,
    zero_division=0
)

recall = recall_score(
    actual,
    predicted,
    zero_division=0
)

f1 = f1_score(
    actual,
    predicted,
    zero_division=0
)


# Confusion matrix
tn, fp, fn, tp = confusion_matrix(
    actual,
    predicted
).ravel()


# False-positive rate
false_positive_rate = fp / (fp + tn)

# False-negative rate
false_negative_rate = fn / (fn + tp)


# Display results
print("\n" + "=" * 60)
print("RULE ENGINE EVALUATION")
print("=" * 60)

print("\nPrecision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))

print("\nConfusion Matrix:")
print("True Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives  :", tp)

print("\nFalse-positive rate:", round(false_positive_rate, 4))
print("False-negative rate:", round(false_negative_rate, 4))


# Save summary
summary = pd.DataFrame([
    {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "False_Positive_Rate": false_positive_rate,
        "False_Negative_Rate": false_negative_rate,
        "True_Negatives": tn,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Positives": tp
    }
])


output_file = (
    base_dir
    / "experiments"
    / "rule_engine_metrics.csv"
)

summary.to_csv(output_file, index=False)

print("\nMetrics saved to:")
print(output_file)