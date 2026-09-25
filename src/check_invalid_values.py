from pathlib import Path
import pandas as pd


# Project paths
base_dir = Path(__file__).resolve().parent.parent

train_file = base_dir / "data" / "processed" / "train.csv"


# Load training data
print("Loading training data...")

df = pd.read_csv(train_file)

df.columns = df.columns.str.strip()

print("Rows:", len(df))
print("Columns:", len(df.columns))


# Numeric features only
numeric_df = df.select_dtypes(include=["number"])


print("\n" + "=" * 70)
print("NEGATIVE VALUE ANALYSIS")
print("=" * 70)


results = []


for column in numeric_df.columns:

    negative_count = (numeric_df[column] < 0).sum()

    if negative_count > 0:

        benign_count = (
            (df["Label"] == "BENIGN")
            & (df[column] < 0)
        ).sum()

        ddos_count = (
            (df["Label"] == "DDoS")
            & (df[column] < 0)
        ).sum()

        results.append({
            "Feature": column,
            "Negative_Count": negative_count,
            "Percentage": (negative_count / len(df)) * 100,
            "BENIGN": benign_count,
            "DDoS": ddos_count,
            "Minimum": numeric_df[column].min()
        })


# Display results
if results:

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "Negative_Count",
        ascending=False
    )

    print("\nFeatures containing negative values:\n")
    print(
        results_df.to_string(
            index=False,
            formatters={
                "Percentage": "{:.4f}".format,
                "Minimum": "{:.4f}".format
            }
        )
    )

else:

    results_df = pd.DataFrame()

    print("\nNo negative values found.")


# Overall negative cells
total_negative_cells = (
    numeric_df < 0
).sum().sum()


print("\n" + "=" * 70)
print("OVERALL SUMMARY")
print("=" * 70)

print("Total negative numeric cells:", total_negative_cells)


# Save results
output_file = (
    base_dir
    / "experiments"
    / "invalid_value_analysis.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print("\nAnalysis saved to:")
print(output_file)