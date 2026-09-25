from pathlib import Path
import pandas as pd


base_dir = Path(__file__).resolve().parent.parent
train_file = base_dir / "data" / "processed" / "train.csv"


print("Loading training data...")

df = pd.read_csv(train_file)
df.columns = df.columns.str.strip()


# Find the 19 suspicious rows
mask = df["Fwd Header Length"] < -1000
suspicious = df.loc[mask].copy()


print("\n" + "=" * 70)
print("CORRUPTED ROW ANALYSIS")
print("=" * 70)

print("\nSuspicious rows:", len(suspicious))


# Check all numeric columns for extreme values
numeric_columns = suspicious.select_dtypes(
    include=["int64", "float64"]
).columns


print("\nExtreme values found in suspicious rows:")
print("-" * 70)

for column in numeric_columns:

    values = suspicious[column]

    negative_count = (values < -1000).sum()
    very_large_count = (values > 1_000_000_000).sum()

    if negative_count > 0 or very_large_count > 0:

        print(
            f"{column}: "
            f"negative extreme = {negative_count}, "
            f"very large = {very_large_count}"
        )


# Compare suspicious rows with normal BENIGN rows
benign = df[df["Label"] == "BENIGN"]

print("\n" + "=" * 70)
print("CLASS BALANCE IMPACT")
print("=" * 70)

print("\nBENIGN rows before removing suspicious rows:")
print(len(benign))

print("\nSuspicious BENIGN rows:")
print(len(suspicious))

print("\nBENIGN rows after removing suspicious rows:")
print(len(benign) - len(suspicious))


# Overall dataset
print("\n" + "=" * 70)
print("OVERALL DATASET")
print("=" * 70)

print("\nTotal training rows:", len(df))
print("Suspicious rows:", len(suspicious))
print(
    "Percentage suspicious:",
    round(len(suspicious) / len(df) * 100, 4),
    "%"
)


# Save only an analysis summary
summary_file = (
    base_dir
    / "experiments"
    / "corrupted_row_analysis.txt"
)

with open(summary_file, "w", encoding="utf-8") as f:

    f.write("CORRUPTED ROW ANALYSIS\n")
    f.write("======================\n\n")

    f.write(f"Suspicious rows: {len(suspicious)}\n")
    f.write(f"Total training rows: {len(df)}\n")
    f.write(
        f"Percentage suspicious: "
        f"{len(suspicious) / len(df) * 100:.4f}%\n\n"
    )

    f.write("Labels:\n")
    f.write(
        suspicious["Label"]
        .value_counts()
        .to_string()
    )


print("\nAnalysis saved to:")
print(summary_file)