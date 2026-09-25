Cybersecurity Detective

An intelligent network threat detection and investigation platform combining supervised machine learning, unsupervised anomaly detection, rule-based evidence, risk scoring, and explainable threat analysis.

Overview

Cybersecurity Detective analyzes network-flow data to identify suspicious activity and support security investigation.

The system combines three complementary detection signals:

Machine Learning Detection — Random Forest classification of network flows.
Anomaly Detection — Isolation Forest trained on benign traffic to identify unusual behavior.
Rule-Based Evidence — Network indicators that provide additional contextual evidence.

These signals are combined into a risk score and threat level, followed by evidence generation and investigation storage.

System Architecture
Network Traffic / Dataset
          |
          v
   Data Preprocessing
          |
          v
   Feature Engineering
          |
     +----+----+
     |         |
     v         v
 ML Detector  Anomaly Detector
     |         |
     +----+----+
          |
          v
      Rule Engine
          |
          v
     Risk Scoring
          |
          v
 Threat Classification
          |
          v
 Evidence Generation
          |
          v
    Investigation DB
          |
     +----+----+
     |         |
     v         v
 REST API   Dashboard
Key Features
Network-flow threat classification
Random Forest supervised learning
Isolation Forest anomaly detection
Benign-only anomaly-model training
Rule-based traffic evidence
Risk scoring
Threat levels: LOW, MEDIUM, HIGH, CRITICAL
Explainable evidence generation
SQLite investigation database
REST API
Investigation dashboard
Strict feature-schema validation
Automated API and integration tests
Detection latency benchmarking
Temporal evaluation
Cross-day evaluation
Feature-ablation experiments
Reproducible preprocessing pipeline
Docker configuration
Structured application logging
Dataset

The project uses the CIC-IDS2017 dataset from the Canadian Institute for Cybersecurity.

The development dataset was constructed from network-flow CSV files containing benign and DDoS traffic.

The project uses flow-level data rather than raw packet captures.

Dataset source:

https://www.unb.ca/cic/datasets/ids-2017.html

Data Preparation

The initial dataset contained:

79 columns
78 numerical features
1 target label

The preprocessing workflow included:

Loading network-flow CSV files.
Removing missing and infinite values.
Detecting duplicate records.
Investigating suspicious negative values.
Identifying sentinel values.
Removing confirmed corrupted flow records.
Removing constant features.
Creating a controlled training/test split.
Building a reproducible preprocessing pipeline.
Applying median imputation and feature scaling.

The final modeling dataset contains 68 numerical features.

Detailed methodology is documented in:

docs/data_cleaning.md

Machine Learning
Supervised Detection

Several baseline classifiers were evaluated:

Logistic Regression
Decision Tree
Random Forest

Random Forest was selected for the production detection component.

The saved model is loaded through:

src/ml_detector.py

The detector validates the incoming feature schema before prediction to prevent missing or unexpected features from silently producing incorrect results.

Development Evaluation

Controlled random-split evaluation produced approximately:

Model	Accuracy	Precision	Recall	F1
Logistic Regression	0.9913	0.9841	0.9989	0.9914
Decision Tree	0.9999	0.9998	0.9999	0.9999
Random Forest	0.9999	0.9999	0.9999	0.9999

These results are development-split measurements and should not be interpreted as general real-world performance.

Temporal Evaluation

Because random train/test splits can overestimate performance on network-traffic datasets, a temporal evaluation was also performed.

The final temporal evaluation used traffic from a later portion of the Friday dataset as the test period.

Results:

Accuracy: 0.9983
Precision: 0.9970
Recall: 0.9989
F1: 0.9979
False positives: 81
False negatives: 30

A separate Monday benign-traffic evaluation was also performed to examine false-positive behavior on traffic from another day.

The Monday evaluation produced:

Benign flows evaluated: 502,650
False positives: 604
False-positive rate: 0.001202

These results are dataset-specific and do not establish performance on arbitrary production networks.

Anomaly Detection

The anomaly detector uses Isolation Forest trained only on benign traffic.

This design treats benign traffic as the reference distribution and asks whether a new flow is sufficiently unusual relative to that baseline.

Model comparison

An earlier mixed-data Isolation Forest was compared with the benign-only approach.

Model	Accuracy	Precision	Recall	F1
Mixed-data model	0.4450	0.2498	0.0501	0.0835
Benign-only model	0.6038	0.8018	0.2849	0.4204

The comparison is based on the project's evaluation split and is not a universal claim about Isolation Forest.

Detailed results:

experiments/isolation_forest_model_comparison.md

Rule Engine

The rule engine provides additional traffic evidence.

Current rules include indicators such as:

High packet rate
High byte rate
Burst traffic
High forward traffic volume

Rules are treated as supporting evidence, rather than a standalone replacement for the ML detector.

This is important because simple fixed thresholds can behave differently across datasets and network environments.

Risk Scoring

The detection pipeline combines multiple signals.

Current scoring components:

ML signal: up to 60 points
Anomaly signal: up to 20 points
Rule evidence: up to 20 points

Threat levels:

Risk Score	Threat Level
0–29.99	LOW
30–59.99	MEDIUM
60–79.99	HIGH
80–100	CRITICAL

The risk calculation is implemented in:

src/risk_scoring.py

Investigation Pipeline

A complete /detect request follows this flow:

Input Network Flow
       |
       v
Feature Validation
       |
       +----------------+
       |                |
       v                v
Random Forest     Isolation Forest
       |                |
       +-------+--------+
               |
               v
          Rule Engine
               |
               v
          Risk Scoring
               |
               v
       Evidence Generation
               |
               v
      Investigation Storage
               |
               v
            API
REST API

The Flask API is implemented in:

src/api.py

Health Check
GET /health

Returns the API health status.

Machine Learning Prediction
POST /predict

Runs the supervised ML detector.

Full Detection
POST /detect

Runs the integrated detection pipeline:

ML prediction
anomaly detection
rule evaluation
risk scoring
evidence generation
database storage
Investigation History
GET /investigations

Returns recent stored investigations.

Example Detection Response
{
  "prediction": "SUSPICIOUS",
  "ml_prediction": "DDoS",
  "ml_probability": 1.0,
  "anomaly_score": 0.5436,
  "rule_alerts": [],
  "risk": {
    "risk_score": 70.87,
    "threat_level": "HIGH"
  }
}

Values shown above are examples from local development testing.

Dashboard

The project includes a Flask-based investigation dashboard.

The dashboard provides access to:

Recent investigations
Prediction results
Risk scores
Threat levels
Investigation details
Generated evidence

Dashboard entry point:

src/dashboard.py

Database

Investigation results are stored in SQLite during local development.

Database module:

src/database.py

Stored information includes fields such as:

Timestamp
Source IP
Destination IP
Source port
Destination port
Protocol
ML prediction
ML probability
Anomaly score
Risk score
Threat level
Evidence

The local database file is intentionally excluded from Git.

Testing

The project uses pytest.

The test suite covers:

Core components
API endpoints
Feature validation
ML detector loading
ML predictions
Anomaly detection
Detection pipeline integration
Missing-feature rejection
Unexpected-feature rejection

Current regression result:

20 tests passed

Run the complete test suite with:

pytest tests -v
Performance Benchmark

A local detection benchmark was performed using 100 network flows.

Results:

Metric	Result
Mean latency	167.401 ms
Median latency	106.783 ms
P95 latency	397.644 ms
P99 latency	501.887 ms
Minimum	100.582 ms
Maximum	812.771 ms
Throughput	5.97 flows/sec

These measurements were collected on the development machine and should not be treated as production infrastructure benchmarks.

Detailed results:

experiments/detection_latency_benchmark.txt

Research Experiments

The project includes several controlled experiments:

Logistic Regression baseline
Decision Tree evaluation
Random Forest evaluation
Feature importance analysis
Isolation Forest contamination analysis
Benign-only anomaly detection
Temporal validation
Cross-day evaluation
Feature ablation
Rule-engine evaluation
Detection latency benchmarking

Experiment outputs are stored in:

experiments/

Feature Ablation

The effect of reducing the feature set was evaluated using the most important features.

The experiment compared:

Top 68 features
Top 50 features
Top 30 features
Top 20 features
Top 10 features

The results are stored in:

experiments/feature_ablation.csv

The project retains the complete 68-feature representation for the current production pipeline because reducing the feature set changed temporal false-positive behavior.

Project Structure
CybersecurityDetective/
│
├── data/
│   ├── processed/
│   │   ├── trained models
│   │   ├── preprocessing artifacts
│   │   └── processed datasets
│   └── cybersecurity.db
│
├── docs/
│   └── data_cleaning.md
│
├── experiments/
│   ├── model evaluation results
│   ├── anomaly experiments
│   ├── temporal evaluation
│   ├── feature ablation
│   ├── rule evaluation
│   └── performance benchmarks
│
├── src/
│   ├── api.py
│   ├── dashboard.py
│   ├── database.py
│   ├── detection_pipeline.py
│   ├── ml_detector.py
│   ├── anomaly_signal.py
│   ├── rule_engine.py
│   ├── risk_scoring.py
│   ├── evidence_generator.py
│   └── training/evaluation scripts
│
├── templates/
│   ├── dashboard.html
│   └── investigation.html
│
├── tests/
│   ├── test_core.py
│   ├── test_api.py
│   └── test_detection_pipeline.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
Local Setup
1. Clone the repository
git clone <repository-url>
cd CybersecurityDetective
2. Create a virtual environment
python -m venv .venv

Activate it:

.venv\Scripts\Activate.ps1
3. Install dependencies
pip install -r requirements.txt
4. Run the API
python src/api.py

The API runs on:

http://127.0.0.1:5000
5. Run the dashboard

Open another terminal and run:

python src/dashboard.py

The dashboard runs on:

http://127.0.0.1:5001
6. Run tests
pytest tests -v
Docker

A Docker configuration is included for reproducible deployment:

Dockerfile
.dockerignore

The current container configuration uses Python 3.14 and installs the pinned project dependencies.

Docker deployment has not been used for the local benchmark because the development environment does not currently have Docker installed.

Engineering Practices

The project follows several software-engineering practices:

Modular Python architecture
Separation of detection components
Reusable preprocessing pipeline
Strict feature-schema validation
Automated tests
Integration testing
Structured logging
Database persistence
Reproducible model artifacts
Controlled experiments
Temporal validation
Performance benchmarking
Git version control
Docker configuration
Limitations

The current system has several important limitations.

Dataset limitation

CIC-IDS2017 is a research dataset and does not represent every real-world network environment.

Generalization

High performance on the evaluated dataset does not guarantee equivalent performance on unseen enterprise networks.

Temporal evaluation

The current temporal evaluation provides stronger evidence than a purely random split, but broader evaluation across multiple attack types and unseen days is still required.

Rule engine

The current rules are evidence generators and are not sufficiently robust to serve as a standalone detection system.

Anomaly detection

Isolation Forest anomaly scores depend on the reference traffic distribution and require further calibration for different network environments.

Production deployment

The current implementation is a research/portfolio system rather than a hardened production IDS.

Future Work

Planned improvements include:

Evaluation on additional CIC-IDS2017 attack categories
Stronger unseen-day evaluation
Additional datasets
False-positive investigation
Error analysis
Model calibration
Anomaly-threshold optimization
Rule-engine refinement
PostgreSQL deployment
Containerized deployment
CI/CD
Cloud deployment
Monitoring and observability
Authentication and authorization
Real-time network-flow ingestion
Streaming detection
Alert notification
More advanced explainability
Research-grade statistical comparisons
Research Direction

The project is designed to evolve from a portfolio implementation into a research-oriented cybersecurity system.

Potential research questions include:

How does combining supervised detection with benign-baseline anomaly detection affect detection robustness?
How does temporal validation change conclusions compared with random train/test evaluation?
How much feature reduction can be achieved without significantly increasing false positives?
Can calibrated multi-signal risk scoring improve investigation prioritization?
How do detection models generalize across different network-traffic distributions?
Reproducibility

Experiments use fixed random seeds where appropriate.

Model preprocessing and inference artifacts are saved separately so that training-time transformations can be reproduced during inference.

Experiment outputs are preserved under:

experiments/

Dataset and model artifacts are excluded from Git when appropriate because of their size.

Disclaimer

This project is intended for educational, research, and defensive cybersecurity purposes.

It analyzes network-flow data and does not provide instructions for compromising systems or conducting unauthorized attacks.

Author

Sudheeksha

Computer Science and Engineering — Artificial Intelligence & Machine Learning

Project status: Active research and engineering project