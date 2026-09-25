from pathlib import Path
import pandas as pd


base_dir = Path(__file__).resolve().parent.parent

clean_file = base_dir / "data" / "processed" / "train_clean.csv"


print("Loading cleaned training data...")

df = pd.read_csv(clean_file)

df.columns = df.columns.str.strip()


print("\n" + "=" * 70)
print("CLEAN DATA VERIFICATION")
print("=" * 70)


print("\nRows:", len(df))
print("Columns:", len(df.columns))


# Check extreme corrupted header values
header_columns = [
    "Fwd Header Length",
    "Fwd Header Length.1",
    "min_seg_size_forward",
    "Bwd Header Length"
]


print("\nExtreme negative values (< -1000):")

total_extreme = 0

for column in header_columns:

    count = (df[column] < -1000).sum()

    print(f"{column}: {count}")

    total_extreme += count


print("\nTotal extreme values:", total_extreme)


# Check labels
print("\nClass distribution:")
print(df["Label"].value_counts())


# Check missing values
missing = df.isna().sum().sum()

print("\nTotal missing values:", missing)


# Check infinite values
numeric_df = df.select_dtypes(include=["int64", "float64"])

infinite_count = (
    numeric_df.isin([float("inf"), float("-inf")])
    .sum()
    .sum()
)

print("Total infinite values:", infinite_count)


if total_extreme == 0 and missing == 0 and infinite_count == 0:
    print("\n✓ Clean dataset verification PASSED")
else:
    print("\n⚠ Further data cleaning is required")