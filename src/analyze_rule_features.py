from pathlib import Path
import pandas as pd


# Project paths
base_dir = Path(__file__).resolve().parent.parent

train_file = base_dir / "data" / "processed" / "train.csv"


# Load training data only
print("Loading training data...")

df = pd.read_csv(train_file)

# Clean column names
df.columns = df.columns.str.strip()

print("Rows:", len(df))


# Features to investigate
features = [
    "Flow Packets/s",
    "Flow Bytes/s",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Duration"
]


print("\n" + "=" * 70)
print("FEATURE DISTRIBUTION ANALYSIS")
print("=" * 70)


results = []


for feature in features:

    print("\n" + "-" * 70)
    print("FEATURE:", feature)
    print("-" * 70)

    benign = df.loc[
        df["Label"] == "BENIGN",
        feature
    ]

    ddos = df.loc[
        df["Label"] == "DDoS",
        feature
    ]

    print("\nBENIGN:")
    print(benign.describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ))

    print("\nDDoS:")
    print(ddos.describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ))


    # Store useful percentile information
    results.append({
        "Feature": feature,

        "BENIGN_50": benign.median(),
        "BENIGN_90": benign.quantile(0.90),
        "BENIGN_95": benign.quantile(0.95),
        "BENIGN_99": benign.quantile(0.99),

        "DDoS_50": ddos.median(),
        "DDoS_90": ddos.quantile(0.90),
        "DDoS_95": ddos.quantile(0.95),
        "DDoS_99": ddos.quantile(0.99)
    })


# Save summary
summary = pd.DataFrame(results)

output_file = (
    base_dir
    / "experiments"
    / "rule_feature_analysis.csv"
)

summary.to_csv(output_file, index=False)


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(summary.to_string(index=False))

print("\nAnalysis saved to:")
print(output_file)