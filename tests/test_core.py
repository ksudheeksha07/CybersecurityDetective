import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


from risk_scoring import calculate_risk_score
from evidence_generator import generate_evidence


def test_risk_score_range():
    result = calculate_risk_score(
        ml_probability=0.92,
        anomaly_score=0.70,
        rule_alerts=[
            "HIGH_PACKET_RATE",
            "HIGH_BYTE_RATE"
        ]
    )

    assert 0 <= result["risk_score"] <= 100


def test_high_threat_level():
    result = calculate_risk_score(
        ml_probability=0.95,
        anomaly_score=0.90,
        rule_alerts=[
            "HIGH_PACKET_RATE",
            "HIGH_BYTE_RATE"
        ]
    )

    assert result["threat_level"] == "CRITICAL"


def test_low_threat_level():
    result = calculate_risk_score(
        ml_probability=0.10,
        anomaly_score=0.05,
        rule_alerts=[]
    )

    assert result["threat_level"] == "LOW"


def test_rule_points():
    result = calculate_risk_score(
        ml_probability=0.50,
        anomaly_score=0.50,
        rule_alerts=[
            "RULE_A",
            "RULE_B"
        ]
    )

    assert result["rule_points"] == 10


def test_evidence_generation():
    evidence = generate_evidence(
        ml_probability=0.92,
        anomaly_score=0.70,
        rule_alerts=[
            "HIGH_PACKET_RATE"
        ],
        feature_values={
            "Flow Packets/s": 8000
        }
    )

    assert len(evidence) > 0
    assert any(
        "HIGH_PACKET_RATE" in item
        for item in evidence
    )


def test_benign_evidence():
    evidence = generate_evidence(
        ml_probability=0.10,
        anomaly_score=0.05,
        rule_alerts=[],
        feature_values={}
    )

    assert len(evidence) > 0