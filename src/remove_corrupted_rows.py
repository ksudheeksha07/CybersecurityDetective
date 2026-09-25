from pathlib import Path
import pandas as pd


base_dir = Path(__file__).resolve().parent.parent

train_file = base_dir / "data" / "processed" / "train.csv"
output_file = base_dir / "data" / "processed" / "train_clean.csv"


print("Loading training data...")

df = pd.read_csv(train_file)

df.columns = df.columns.str.strip()


# Identify confirmed corrupted rows
mask = df["Fwd Header Length"] < -1000

corrupted_count = mask.sum()


print("\nConfirmed corrupted rows:", corrupted_count)


# Remove only those rows
clean_df = df.loc[~mask].copy()


print("Original training rows:", len(df))
print("Clean training rows:", len(clean_df))
print("Rows removed:", len(df) - len(clean_df))


# Check class distribution
print("\nClass distribution before:")
print(df["Label"].value_counts())


print("\nClass distribution after:")
print(clean_df["Label"].value_counts())


# Save new dataset
clean_df.to_csv(output_file, index=False)


print("\nClean training dataset saved to:")
print(output_file)