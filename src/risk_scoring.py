def calculate_risk_score(
    ml_probability,
    anomaly_score=0.0,
    rule_alerts=None
):
    """
    Calculate a defensive cybersecurity risk score from 0 to 100.

    Components:
    - ML confidence: up to 60 points
    - Anomaly signal: up to 20 points
    - Rule evidence: up to 20 points
    """

    if rule_alerts is None:
        rule_alerts = []

    # ML contribution
    ml_score = max(0.0, min(1.0, ml_probability)) * 60

    # Convert anomaly score to a bounded 0-20 contribution
    anomaly_score = max(0.0, min(1.0, anomaly_score))
    anomaly_points = anomaly_score * 20

    # Rule evidence
    rule_points = min(len(rule_alerts) * 5, 20)

    # Final risk score
    risk_score = ml_score + anomaly_points + rule_points
    risk_score = round(min(risk_score, 100), 2)

    # Threat level
    if risk_score >= 80:
        threat_level = "CRITICAL"
    elif risk_score >= 60:
        threat_level = "HIGH"
    elif risk_score >= 30:
        threat_level = "MEDIUM"
    else:
        threat_level = "LOW"

    return {
        "risk_score": risk_score,
        "threat_level": threat_level,
        "ml_points": round(ml_score, 2),
        "anomaly_points": round(anomaly_points, 2),
        "rule_points": round(rule_points, 2),
        "evidence": rule_alerts
    }


if __name__ == "__main__":
    result = calculate_risk_score(
        ml_probability=0.92,
        anomaly_score=0.70,
        rule_alerts=[
            "HIGH_PACKET_RATE",
            "HIGH_BYTE_RATE"
        ]
    )

    print("Risk Assessment")
    print("-" * 40)

    for key, value in result.items():
        print(f"{key}: {value}")