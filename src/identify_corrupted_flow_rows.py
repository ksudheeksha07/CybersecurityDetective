from pathlib import Path
import pandas as pd


base_dir = Path(__file__).resolve().parent.parent

train_file = base_dir / "data" / "processed" / "train_clean.csv"


print("Loading cleaned training data...")

df = pd.read_csv(train_file)

df.columns = df.columns.str.strip()


features = [
    "Flow IAT Min",
    "Flow Duration",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Max",
    "Fwd IAT Min",
    "Flow Bytes/s"
]


# Count invalid negative values in each row
negative_count = pd.Series(
    0,
    index=df.index
)

for feature in features:

    negative_count += (
        df[feature] < 0
    ).astype(int)


# Clearly corrupted:
# 3 or more related timing/rate features are negative
corrupted_mask = negative_count >= 3

corrupted = df.loc[corrupted_mask].copy()

corrupted["Negative_Feature_Count"] = (
    negative_count.loc[corrupted_mask]
)


print("\n" + "=" * 70)
print("CLEARLY CORRUPTED FLOW ROWS")
print("=" * 70)


print("\nNumber of rows:", len(corrupted))


print("\nLabels:")
print(corrupted["Label"].value_counts())


print("\nRows and invalid feature counts:")

print(
    corrupted[
        ["Label", "Negative_Feature_Count"] + features
    ].to_string(index=False)
)


print("\nClass distribution if removed:")

print("\nBefore:")
print(df["Label"].value_counts())


clean_df = df.loc[~corrupted_mask]

print("\nAfter:")
print(clean_df["Label"].value_counts())


print("\nRows removed:", len(corrupted))
print(
    "Percentage of dataset:",
    round(len(corrupted) / len(df) * 100, 4),
    "%"
)


# Save analysis only
output_file = (
    base_dir
    / "experiments"
    / "clearly_corrupted_flow_rows.csv"
)

corrupted.to_csv(
    output_file,
    index=False
)


print("\nAnalysis saved to:")
print(output_file)