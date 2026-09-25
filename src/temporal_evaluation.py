from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# Paths
base_dir = Path(__file__).resolve().parent.parent

data_file = (
    Path.home()
    / "Downloads"
    / "MachineLearningCSV"
    / "MachineLearningCVE"
    / "selected"
    / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)


# Read the Friday dataset
print("Loading Friday dataset...")

df = pd.read_csv(data_file)

print("Original shape:", df.shape)


# Clean column names
df.columns = df.columns.str.strip()


# Replace infinite values
df = df.replace([float("inf"), float("-inf")], pd.NA)

# Remove rows with missing values
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()

print("Cleaned shape:", df.shape)


# Separate labels
X = df.drop(columns=["Label"])
y = df["Label"]


# Keep only numeric features
X = X.select_dtypes(include=["number"])


# Remove constant features
constant_columns = X.columns[X.nunique() <= 1]
X = X.drop(columns=constant_columns)

print("Features used:", X.shape[1])


# Split chronologically
split_index = int(len(df) * 0.70)

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining labels:")
print(y_train.value_counts())

print("\nTesting labels:")
print(y_test.value_counts())


# Train Random Forest
print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# Evaluate
predictions = model.predict(X_test)

print("\n" + "=" * 60)
print("TEMPORAL EVALUATION RESULTS")
print("=" * 60)

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))

print("\nClassification Report:")
print(classification_report(y_test, predictions))