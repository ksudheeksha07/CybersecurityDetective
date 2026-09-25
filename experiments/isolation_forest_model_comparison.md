# Isolation Forest Model Comparison

## Objective

Compare the original mixed-data Isolation Forest with a BENIGN-only Isolation Forest for anomaly detection.

## Evaluation Dataset

- Dataset: CIC-IDS2017
- Evaluation split: data/processed/test.csv
- Samples: 39,653
- BENIGN: 19,654
- DDoS: 19,999

## Model A — Original Mixed-data Model

- Training data contained both BENIGN and DDoS flows.
- Isolation Forest contamination: 0.50.

Results:

- Accuracy: 0.4450
- Precision: 0.2498
- Recall: 0.0501
- F1: 0.0835
- BENIGN false positives: 3,010
- DDoS detected: 1,002

## Model B — BENIGN-only Model

- Training data contained BENIGN flows only.
- BENIGN training samples: 78,613
- Isolation Forest contamination: auto
- Sentinel -1 values were treated as missing and median-imputed.

Results:

- Accuracy: 0.6038
- Precision: 0.8018
- Recall: 0.2849
- F1: 0.4204
- BENIGN false positives: 1,408
- DDoS detected: 5,697

## Interpretation

The BENIGN-only model produced higher accuracy, precision, recall, and F1 on this evaluation split. It also produced fewer BENIGN false positives while identifying more DDoS flows as anomalous.

The BENIGN-only model will therefore be treated as the candidate anomaly detector for further integration testing.

This result does not establish generalization to other datasets, attack types, or production traffic. Further temporal and unseen-attack evaluation is required.

## Limitations

- Evaluation uses one test split.
- CIC-IDS2017 is a controlled benchmark dataset.
- Only DDoS is evaluated as the attack class in this comparison.
- The anomaly detector is an investigation signal, not the standalone final classifier.
- Further temporal and cross-day validation is required.
