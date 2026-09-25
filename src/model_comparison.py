from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


base_dir = Path(__file__).resolve().parent.parent

model_data_file = base_dir / "data" / "processed" / "model_data.pkl"

print("Loading prepared model data...")

X_train, X_test, y_train, y_test = joblib.load(model_data_file)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}

results = []

for name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", name)
    print("=" * 60)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        pos_label="DDoS"
    )
    recall = recall_score(
        y_test,
        y_pred,
        pos_label="DDoS"
    )
    f1 = f1_score(
        y_test,
        y_pred,
        pos_label="DDoS"
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=["BENIGN", "DDoS"]
    ).ravel()

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "False Positives": fp,
        "False Negatives": fn
    })

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1:", round(f1, 4))
    print("False Positives:", fp)
    print("False Negatives:", fn)

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

output_file = (
    base_dir
    / "experiments"
    / "model_comparison.csv"
)

results_df.to_csv(output_file, index=False)

print("\nComparison saved to:")
print(output_file)

print("\n✓ MODEL COMPARISON COMPLETE")