from pathlib import Path
import pandas as pd
import numpy as np

base_dir = Path(__file__).resolve().parent.parent
file = base_dir / "data" / "processed" / "train_final.csv"

print("Loading final training dataset...")

df = pd.read_csv(file)
df.columns = df.columns.str.strip()

print("\n" + "=" * 70)
print("FINAL DATASET VERIFICATION")
print("=" * 70)

print("\nRows:", len(df))
print("Columns:", len(df.columns))

print("\nMissing values:", df.isna().sum().sum())
print("Infinite values:", np.isinf(df.select_dtypes(include=np.number)).sum().sum())

print("\nClass distribution:")
print(df["Label"].value_counts())

print("\nRemaining negative values:")

negative_found = False

for column in df.select_dtypes(include=np.number).columns:
    count = (df[column] < 0).sum()

    if count > 0:
        negative_found = True
        print(f"{column}: {count}")

if not negative_found:
    print("No negative values found.")

print("\nDuplicate rows:", df.duplicated().sum())

print("\n" + "=" * 70)

if (
    df.isna().sum().sum() == 0
    and np.isinf(df.select_dtypes(include=np.number)).sum().sum() == 0
    and df.duplicated().sum() == 0
):
    print("✓ FINAL DATASET VERIFICATION PASSED")
else:
    print("⚠ REVIEW REQUIRED")

print("=" * 70)