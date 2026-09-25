try:
    from src.ml_detector import MLDetector
    from src.anomaly_signal import AnomalyDetector
    from src.rule_engine import evaluate_rules
    from src.risk_scoring import calculate_risk_score
    from src.evidence_generator import generate_evidence

except ModuleNotFoundError:
    from ml_detector import MLDetector
    from anomaly_signal import AnomalyDetector
    from rule_engine import evaluate_rules
    from risk_scoring import calculate_risk_score
    from evidence_generator import generate_evidence


class DetectionPipeline:
    """
    Unified defensive network-threat detection pipeline.

    Combines:
    1. Supervised ML classification
    2. Unsupervised anomaly detection
    3. Rule-based evidence
    4. Risk scoring
    5. Explainable evidence
    """

    def __init__(self):
        self.ml_detector = MLDetector()
        self.anomaly_detector = AnomalyDetector()

    def analyze(self, features):
        """
        Analyze a single network flow.

        Parameters
        ----------
        features : dict
            Network-flow features matching the trained ML model schema.

        Returns
        -------
        dict
            Complete detection result.
        """

        # 1. Supervised ML prediction
        ml_result = self.ml_detector.predict(features)

        ml_probability = ml_result["ml_probability"]
        ml_prediction = ml_result["prediction"]

        # 2. Anomaly detection
        anomaly_scores = self.anomaly_detector.get_anomaly_signal(
            features
        )

        anomaly_score = float(anomaly_scores[0])

        # 3. Rule-based evidence
        rule_result = evaluate_rules(features)

        rule_alerts = rule_result["triggered_rules"]

        # 4. Combined risk score
        risk = calculate_risk_score(
            ml_probability=ml_probability,
            anomaly_score=anomaly_score,
            rule_alerts=rule_alerts
        )

        # 5. Explainable evidence
        evidence = generate_evidence(
            ml_probability=ml_probability,
            anomaly_score=anomaly_score,
            rule_alerts=rule_alerts,
            feature_values=features
        )

        # 6. Final investigation classification
        prediction = (
            "SUSPICIOUS"
            if ml_probability >= 0.5
            else "BENIGN"
        )

        return {
            "prediction": prediction,
            "ml_prediction": ml_prediction,
            "ml_probability": ml_probability,
            "anomaly_score": anomaly_score,
            "rule_alerts": rule_alerts,
            "risk": risk,
            "evidence": evidence
        }


if __name__ == "__main__":

    import pandas as pd

    test_file = "data/processed/test.csv"

    df = pd.read_csv(
        test_file,
        nrows=1
    )

    df.columns = df.columns.str.strip()

    features = (
        df.drop(columns=["Label"])
        .iloc[0]
        .to_dict()
    )

    pipeline = DetectionPipeline()

    result = pipeline.analyze(features)

    print(
        "Unified detection pipeline loaded successfully."
    )
    print("-" * 60)

    for key, value in result.items():
        print(f"{key}: {value}")