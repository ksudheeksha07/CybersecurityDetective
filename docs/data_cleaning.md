# Data Cleaning and Preprocessing

## Dataset

The project uses the CIC-IDS2017 dataset, specifically the MachineLearningCSV flow records.

The initial experimental dataset was constructed from:

- Monday working-hours traffic: BENIGN flows
- Friday afternoon traffic: DDoS and BENIGN flows

For initial model development, a balanced dataset was created containing:

- 100,000 BENIGN flows
- 100,000 DDoS flows

A random seed of 42 was used for reproducibility.

---

## Initial Dataset

The balanced dataset contained:

- 200,000 flows
- 78 network features
- 1 target label column

The target classes were:

- BENIGN
- DDoS

---

## Cleaning Procedure

The following preprocessing operations were performed.

### 1. Infinite Values

Positive and negative infinite values were replaced with missing values.

Rows containing these invalid values were removed because they cannot be reliably used as numerical network-flow measurements.

### 2. Missing Values

Rows containing missing values after the infinite-value replacement were removed.

Only a very small number of records were affected.

### 3. Duplicate Records

Duplicate flow records were removed to avoid giving repeated observations disproportionate influence during model training.

### 4. Constant Features

Features with the same value for every observation were removed because they contain no information for distinguishing the two classes.

Ten constant features were removed.

The resulting feature set contained:

- 68 numerical features
- 1 target label

### 5. Extreme Corrupted Values

A separate investigation was performed to identify physically implausible negative values.

Nineteen records contained clearly corrupted header-length or segment-size values. These records were all BENIGN and represented only 0.012% of the training data.

These records were removed.

A further investigation identified four records with multiple simultaneously invalid timing and rate measurements, including values such as:

- Flow Duration = -1
- Flow Packets/s = -2,000,000
- Flow IAT Mean = -1
- Flow IAT Max = -1
- Flow Bytes/s = -12,000,000

These four records represented only 0.0025% of the cleaned training data and were removed.

### 6. Sentinel Values

Some CIC-IDS2017 fields contain `-1` values that behave as sentinel values rather than ordinary numerical measurements.

The most important examples are:

- `Init_Win_bytes_forward`
- `Init_Win_bytes_backward`

These values were retained rather than deleting the corresponding rows.

They will be handled explicitly during model preprocessing.

---

## Final Training Dataset

After cleaning, the final training dataset contains:

- 158,586 flows
- 68 numerical features
- 1 target label
- 0 missing values
- 0 infinite values
- 0 duplicate rows

Class distribution:

| Class | Samples |
|---|---:|
| DDoS | 79,996 |
| BENIGN | 78,590 |

The dataset is therefore approximately balanced.

---

## Reproducibility

The following random seed was used where applicable:

`random_state = 42`

All original CIC-IDS2017 files remain unchanged.

The processed datasets are stored separately under:

`data/processed/`

This allows the preprocessing pipeline and experimental datasets to be reproduced without modifying the original source data.

---

## Important Evaluation Note

The balanced dataset and random train/test split are used for initial model development.

They should not be treated as the final measure of real-world generalization.

Later experiments will evaluate models on temporally separated or unseen-day traffic to reduce the possibility of overly optimistic results caused by similar flows appearing in both training and testing data.