from pathlib import Path
import pandas as pd


# Project paths
base_dir = Path(__file__).resolve().parent.parent

train_file = base_dir / "data" / "processed" / "train.csv"


# Load training data
print("Loading training data...")

df = pd.read_csv(train_file)

df.columns = df.columns.str.strip()


# Features with suspicious extreme negative values
suspicious_features = [
    "Fwd Header Length",
    "Fwd Header Length.1",
    "min_seg_size_forward",
    "Bwd Header Length"
]


print("\n" + "=" * 70)
print("EXTREME NEGATIVE VALUE INSPECTION")
print("=" * 70)


# Find rows containing extreme negative values
mask = pd.Series(False, index=df.index)

for feature in suspicious_features:

    if feature in df.columns:
        mask = mask | (df[feature] < -1000)


extreme_rows = df.loc[mask].copy()


print("\nNumber of suspicious rows:", len(extreme_rows))


# Show labels
print("\nLabels of suspicious rows:")
print(extreme_rows["Label"].value_counts())


# Show suspicious feature values
print("\nSuspicious feature values:")

columns_to_show = ["Label"] + [
    feature
    for feature in suspicious_features
    if feature in df.columns
]

print(
    extreme_rows[columns_to_show].to_string(
        index=False
    )
)


# Check how many suspicious features each row contains
extreme_rows["Suspicious_Feature_Count"] = 0

for feature in suspicious_features:

    if feature in df.columns:

        extreme_rows["Suspicious_Feature_Count"] += (
            extreme_rows[feature] < -1000
        ).astype(int)


print("\nNumber of suspicious features per row:")

print(
    extreme_rows["Suspicious_Feature_Count"]
    .value_counts()
    .sort_index()
)


# Save inspection results
output_file = (
    base_dir
    / "experiments"
    / "extreme_value_inspection.csv"
)

extreme_rows.to_csv(
    output_file,
    index=False
)


print("\nInspection saved to:")
print(output_file)