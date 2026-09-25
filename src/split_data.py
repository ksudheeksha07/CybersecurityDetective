from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

input_file = Path("data/processed/network_traffic_features.csv")

train_file = Path("data/processed/train.csv")
test_file = Path("data/processed/test.csv")

df = pd.read_csv(input_file)

X = df.drop(columns=[" Label"])
y = df[" Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

train_data = X_train.copy()
train_data[" Label"] = y_train.values

test_data = X_test.copy()
test_data[" Label"] = y_test.values

train_data.to_csv(train_file, index=False)
test_data.to_csv(test_file, index=False)

print("=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print("\nTraining samples:", len(train_data))
print("Testing samples:", len(test_data))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

print("\nSaved:")
print("-", train_file)
print("-", test_file)