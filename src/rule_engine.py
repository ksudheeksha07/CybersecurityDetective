from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "test.csv"
OUTPUT_FILE = BASE_DIR / "experiments" / "rule_engine_results.csv"


RULE_NAMES = [
    "HIGH_PACKET_RATE",
    "HIGH_BYTE_RATE",
    "BURST_TRAFFIC",
    "HIGH_FWD_VOLUME"
]


def apply_rules(row):
    """
    Apply defensive network-traffic rules.

    Returns:
        list[str]: Names of rules triggered by the flow.
    """

    triggered_rules = []

    # Rule 1: Very high packet rate
    if (
        pd.notna(row.get("Flow Packets/s"))
        and row["Flow Packets/s"] > 5000
    ):
        triggered_rules.append("HIGH_PACKET_RATE")

    # Rule 2: Very high byte rate
    if (
        pd.notna(row.get("Flow Bytes/s"))
        and row["Flow Bytes/s"] > 1_000_000
    ):
        triggered_rules.append("HIGH_BYTE_RATE")

    # Rule 3: Very short duration with many packets
    if (
        pd.notna(row.get("Flow Duration"))
        and pd.notna(row.get("Total Fwd Packets"))
        and row["Flow Duration"] < 1_000_000
        and row["Total Fwd Packets"] > 100
    ):
        triggered_rules.append("BURST_TRAFFIC")

    # Rule 4: Large forward packet volume
    if (
        pd.notna(row.get("Total Length of Fwd Packets"))
        and row["Total Length of Fwd Packets"] > 1_000_000
    ):
        triggered_rules.append("HIGH_FWD_VOLUME")

    return triggered_rules


def evaluate_rules(row):
    """
    Evaluate a single network-flow record.

    Returns:
        dict: Rule alerts and alert count.
    """

    triggered_rules = apply_rules(row)

    return {
        "triggered_rules": triggered_rules,
        "rule_count": len(triggered_rules),
        "rule_alert": len(triggered_rules) > 0
    }


def evaluate_dataframe(dataframe):
    """
    Apply the rule engine to a complete DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with rule results.
    """

    df = dataframe.copy()
    df.columns = df.columns.str.strip()

    df["Triggered_Rules"] = df.apply(
        apply_rules,
        axis=1
    )

    df["Rule_Count"] = df["Triggered_Rules"].apply(len)

    df["Rule_Alert"] = df["Rule_Count"] > 0

    return df


def run_evaluation():
    """
    Evaluate the rules against the project test dataset.
    """

    print("Loading test data...")

    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()

    print("Rows:", len(df))

    print("\nApplying detection rules...")

    df = evaluate_dataframe(df)

    print("\n" + "=" * 60)
    print("RULE ENGINE RESULTS")
    print("=" * 60)

    print("\nTotal flows:", len(df))
    print(
        "Flows triggering at least one rule:",
        int(df["Rule_Alert"].sum())
    )
    print(
        "Flows triggering no rules:",
        int((~df["Rule_Alert"]).sum())
    )

    print("\nRule trigger counts:")

    for rule in RULE_NAMES:
        count = df["Triggered_Rules"].apply(
            lambda rules: rule in rules
        ).sum()

        print(f"{rule}: {int(count)}")

    if "Label" in df.columns:

        print("\nRule alerts by actual class:")

        print(
            pd.crosstab(
                df["Label"],
                df["Rule_Alert"]
            )
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df[
        [
            "Label",
            "Rule_Count",
            "Triggered_Rules",
            "Rule_Alert"
        ]
    ].to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nResults saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    run_evaluation()