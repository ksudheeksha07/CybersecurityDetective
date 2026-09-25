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


print("\n" + "=" * 70)
print("TRACE REMAINING NEGATIVE ROWS")
print("=" * 70)


# Find rows containing suspicious negative values
mask = pd.Series(False, index=df.index)

for feature in features:

    mask = mask | (df[feature] < 0)


suspicious = df.loc[mask].copy()


print("\nTotal rows containing at least one suspicious negative:")
print(len(suspicious))


print("\nLabels:")
print(suspicious["Label"].value_counts())


print("\nNegative values by feature:")
print("-" * 70)

for feature in features:

    count = (suspicious[feature] < 0).sum()

    if count > 0:

        print(f"{feature}: {count}")


# Show the actual suspicious values
print("\n" + "=" * 70)
print("ACTUAL SUSPICIOUS VALUES")
print("=" * 70)

columns_to_show = ["Label"] + features

print(
    suspicious[columns_to_show]
    .to_string(index=False)
)


# Check how many rows contain multiple suspicious values
suspicious["Negative_Feature_Count"] = 0

for feature in features:

    suspicious["Negative_Feature_Count"] += (
        suspicious[feature] < 0
    ).astype(int)


print("\n" + "=" * 70)
print("NEGATIVE FEATURES PER ROW")
print("=" * 70)

print(
    suspicious["Negative_Feature_Count"]
    .value_counts()
    .sort_index()
)


# Save complete suspicious rows
output_file = (
    base_dir
    / "experiments"
    / "remaining_negative_rows.csv"
)

suspicious.to_csv(
    output_file,
    index=False
)


print("\nDetailed rows saved to:")
print(output_file)