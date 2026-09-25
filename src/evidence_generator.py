def generate_evidence(
    ml_probability,
    anomaly_score,
    rule_alerts,
    feature_values=None
):
    """
    Generate human-readable defensive evidence
    explaining why a network flow received its risk assessment.
    """

    evidence = []

    # ML evidence
    if ml_probability >= 0.80:
        evidence.append(
            f"ML detector strongly classified the flow as suspicious "
            f"(confidence: {ml_probability:.2f})."
        )
    elif ml_probability >= 0.50:
        evidence.append(
            f"ML detector identified moderate suspicion "
            f"(confidence: {ml_probability:.2f})."
        )

    # Anomaly evidence
    if anomaly_score >= 0.70:
        evidence.append(
            f"Network behavior is highly anomalous "
            f"(anomaly score: {anomaly_score:.2f})."
        )
    elif anomaly_score >= 0.40:
        evidence.append(
            f"Network behavior shows moderate deviation from normal "
            f"(anomaly score: {anomaly_score:.2f})."
        )

    # Rule evidence
    for rule in rule_alerts:
        evidence.append(
            f"Rule triggered: {rule}"
        )

    # Feature evidence
    if feature_values:
        if feature_values.get("Flow Packets/s", 0) > 5000:
            evidence.append(
                "Very high packet rate detected."
            )

        if feature_values.get("Flow Bytes/s", 0) > 1_000_000:
            evidence.append(
                "Very high byte transfer rate detected."
            )

        if feature_values.get("Total Fwd Packets", 0) > 100:
            evidence.append(
                "High number of forward packets detected."
            )

    if not evidence:
        evidence.append(
            "No strong evidence indicators were detected."
        )

    return evidence


if __name__ == "__main__":
    result = generate_evidence(
        ml_probability=0.92,
        anomaly_score=0.70,
        rule_alerts=[
            "HIGH_PACKET_RATE",
            "HIGH_BYTE_RATE"
        ],
        feature_values={
            "Flow Packets/s": 8000,
            "Flow Bytes/s": 2_000_000,
            "Total Fwd Packets": 120
        }
    )

    print("Threat Evidence")
    print("-" * 40)

    for item in result:
        print(f"- {item}")