from pathlib import Path
import pandas as pd


base_dir = Path(__file__).resolve().parent.parent

train_file = base_dir / "data" / "processed" / "train_clean.csv"


print("Loading cleaned training data...")

df = pd.read_csv(train_file)

df.columns = df.columns.str.strip()


print("\n" + "=" * 70)
print("REMAINING NEGATIVE VALUE ANALYSIS")
print("=" * 70)


numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns


results = []


for column in numeric_columns:

    negative_mask = df[column] < 0

    count = negative_mask.sum()

    if count > 0:

        benign_count = (
            df.loc[negative_mask, "Label"] == "BENIGN"
        ).sum()

        ddos_count = (
            df.loc[negative_mask, "Label"] == "DDoS"
        ).sum()

        minimum = df.loc[negative_mask, column].min()

        results.append({
            "Feature": column,
            "Negative_Count": count,
            "Percentage": round(
                count / len(df) * 100,
                4
            ),
            "BENIGN": benign_count,
            "DDoS": ddos_count,
            "Minimum": minimum
        })


results_df = pd.DataFrame(results)


if len(results_df) > 0:

    results_df = results_df.sort_values(
        "Negative_Count",
        ascending=False
    )

    print("\nFeatures containing negative values:")
    print(results_df.to_string(index=False))

else:

    print("\nNo negative values found.")


output_file = (
    base_dir
    / "experiments"
    / "remaining_negative_analysis.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


print("\nAnalysis saved to:")
print(output_file)