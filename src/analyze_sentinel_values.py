from pathlib import Path
import pandas as pd


base_dir = Path(__file__).resolve().parent.parent

train_file = base_dir / "data" / "processed" / "train_clean.csv"


print("Loading cleaned training data...")

df = pd.read_csv(train_file)

df.columns = df.columns.str.strip()


features = [
    "Init_Win_bytes_backward",
    "Init_Win_bytes_forward"
]


print("\n" + "=" * 70)
print("SENTINEL VALUE ANALYSIS")
print("=" * 70)


for feature in features:

    print("\n" + "-" * 70)
    print("FEATURE:", feature)

    values = df[feature]

    sentinel_mask = values == -1

    print("\nTotal rows:", len(df))
    print("Rows with -1:", sentinel_mask.sum())
    print(
        "Percentage:",
        round(sentinel_mask.mean() * 100, 2),
        "%"
    )

    print("\n-1 values by class:")

    print(
        df.loc[sentinel_mask, "Label"]
        .value_counts()
    )

    print("\nNon--1 values by class:")

    print(
        df.loc[~sentinel_mask, "Label"]
        .value_counts()
    )

    print("\nValue statistics excluding -1:")

    non_sentinel = values[~sentinel_mask]

    print(non_sentinel.describe())


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)