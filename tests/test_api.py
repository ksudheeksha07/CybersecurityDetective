import sys
from pathlib import Path

import joblib
import pytest


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------
# Import API
# ---------------------------------------------------------

from api import app


# ---------------------------------------------------------
# Test client fixture
# ---------------------------------------------------------

@pytest.fixture
def client():
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


# ---------------------------------------------------------
# Helper: build valid detection features
# ---------------------------------------------------------

def build_valid_detect_features():
    """
    Build a feature dictionary using the exact 68-feature
    schema expected by the preprocessing pipeline.
    """

    pipeline_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "preprocessing_pipeline.pkl"
    )

    preprocessing_pipeline = joblib.load(pipeline_path)

    feature_names = list(
        preprocessing_pipeline.feature_names_in_
    )

    features = {
        feature_name: 0.0
        for feature_name in feature_names
    }

    return features


# =========================================================
# BASIC API TESTS
# =========================================================

def test_health_endpoint(client):

    response = client.get("/health")

    assert response.status_code == 200

    data = response.get_json()

    assert data["status"] == "healthy"


def test_home_endpoint(client):

    response = client.get("/")

    assert response.status_code == 200

    data = response.get_json()

    assert data["service"] == "Cybersecurity Detective API"
    assert data["status"] == "running"


# =========================================================
# /analyze TESTS
# =========================================================

def test_analyze_valid_request(client):

    payload = {
        "prediction": "DDoS",
        "ml_probability": 0.95,
        "anomaly_score": 0.8,
        "rule_alerts": [
            "HIGH_PACKET_RATE"
        ]
    }

    response = client.post(
        "/analyze",
        json=payload
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "investigation_id" in data
    assert "prediction" in data
    assert "risk" in data
    assert "risk_score" in data["risk"]

    assert data["prediction"] == "SUSPICIOUS"
    assert 0 <= data["risk"]["risk_score"] <= 100


def test_analyze_missing_fields(client):

    payload = {
        "prediction": "DDoS"
    }

    response = client.post(
        "/analyze",
        json=payload
    )

    assert response.status_code == 400

    data = response.get_json()

    assert "error" in data


def test_analyze_invalid_probability(client):

    payload = {
        "prediction": "DDoS",
        "ml_probability": 1.5,
        "anomaly_score": 0.5,
        "rule_alerts": []
    }

    response = client.post(
        "/analyze",
        json=payload
    )

    assert response.status_code == 400

    data = response.get_json()

    assert "error" in data


def test_analyze_invalid_anomaly_score(client):

    payload = {
        "prediction": "DDoS",
        "ml_probability": 0.9,
        "anomaly_score": -0.5,
        "rule_alerts": []
    }

    response = client.post(
        "/analyze",
        json=payload
    )

    assert response.status_code == 400

    data = response.get_json()

    assert "error" in data


# =========================================================
# /investigations TEST
# =========================================================

def test_investigations_endpoint(client):

    response = client.get("/investigations")

    assert response.status_code == 200

    data = response.get_json()

    assert isinstance(data, dict)
    assert "count" in data
    assert "investigations" in data

    assert isinstance(data["count"], int)
    assert isinstance(data["investigations"], list)

    assert data["count"] == len(data["investigations"])


# =========================================================
# /detect TESTS
# =========================================================

def test_detect_valid_request(client):

    payload = {
        "features": build_valid_detect_features()
    }

    response = client.post(
        "/detect",
        json=payload
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "investigation_id" in data
    assert "prediction" in data
    assert "ml_prediction" in data
    assert "ml_probability" in data
    assert "anomaly_score" in data
    assert "risk" in data
    assert "risk_score" in data["risk"]
    assert "evidence" in data

    assert 0 <= data["ml_probability"] <= 1
    assert 0 <= data["anomaly_score"] <= 1
    assert 0 <= data["risk"]["risk_score"] <= 100


def test_detect_missing_features(client):

    features = build_valid_detect_features()

    # Remove one required feature.
    removed_feature = next(iter(features))

    del features[removed_feature]

    payload = {
        "features": features
    }

    response = client.post(
        "/detect",
        json=payload
    )

    assert response.status_code == 400

    data = response.get_json()

    assert "error" in data
    assert "details" in data