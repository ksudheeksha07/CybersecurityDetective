from pathlib import Path
import pandas as pd

input_file = Path("data/processed/network_traffic_clean.csv")
output_file = Path("data/processed/network_traffic_features.csv")

df = pd.read_csv(input_file)

label_column = " Label"

X = df.drop(columns=[label_column])
y = df[label_column]

# Find constant features
constant_features = [
    column for column in X.columns
    if X[column].nunique() <= 1
]

print("=" * 70)
print("FEATURE CLEANING")
print("=" * 70)

print("\nOriginal feature count:", X.shape[1])

print("\nConstant features removed:")
for feature in constant_features:
    print("-", feature)

# Remove constant features
X = X.drop(columns=constant_features)

# Add label back
df_clean = pd.concat([X, y], axis=1)

# Save
df_clean.to_csv(output_file, index=False)

print("\nFinal feature count:", X.shape[1])
print("Total samples:", len(df_clean))

print("\nSaved to:", output_file)