from pathlib import Path
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score


# Project paths
base_dir = Path(__file__).resolve().parent.parent

train_file = base_dir / "data" / "processed" / "train.csv"
test_file = base_dir / "data" / "processed" / "test.csv"


# Load data
print("Loading training and testing data...")

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)


# Separate features and labels
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

X_train = train_df.drop(columns=["Label"])
y_train = train_df["Label"]

X_test = test_df.drop(columns=["Label"])
y_test = test_df["Label"]


# Keep numeric features only
X_train = X_train.select_dtypes(include=["number"])
X_test = X_test[X_train.columns]


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Features:", X_train.shape[1])


# Train Decision Tree
print("\nTraining Decision Tree...")

model = DecisionTreeClassifier(
    random_state=42
)

model.fit(X_train, y_train)


# Predictions
predictions = model.predict(X_test)


# Evaluation
print("\n" + "=" * 60)
print("DECISION TREE RESULTS")
print("=" * 60)

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))

print("\nClassification Report:")
print(classification_report(y_test, predictions))