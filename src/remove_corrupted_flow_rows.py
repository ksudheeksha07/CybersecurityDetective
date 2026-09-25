from pathlib import Path
import pandas as pd

base_dir = Path(__file__).resolve().parent.parent

input_file = base_dir / "data" / "processed" / "train_clean.csv"
output_file = base_dir / "data" / "processed" / "train_final.csv"

print("Loading cleaned training data...")

df = pd.read_csv(input_file)
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

negative_count = pd.Series(0, index=df.index)

for feature in features:
    negative_count += (df[feature] < 0).astype(int)

corrupted_mask = negative_count >= 3

clean_df = df.loc[~corrupted_mask].copy()

print("\n" + "=" * 70)
print("FINAL TRAINING DATA CLEANING")
print("=" * 70)

print("\nOriginal rows:", len(df))
print("Corrupted rows removed:", corrupted_mask.sum())
print("Final rows:", len(clean_df))

print("\nFinal class distribution:")
print(clean_df["Label"].value_counts())

clean_df.to_csv(output_file, index=False)

print("\nFinal dataset saved to:")
print(output_file)