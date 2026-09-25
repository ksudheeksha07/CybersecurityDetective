import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.ml_detector import MLDetector
from src.anomaly_signal import AnomalyDetector
from src.detection_pipeline import DetectionPipeline


@pytest.fixture
def sample_features():

    df = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "test.csv",
        nrows=1
    )

    df.columns = df.columns.str.strip()

    return (
        df.drop(columns=["Label"])
        .iloc[0]
        .to_dict()
    )


def test_ml_detector_loads():

    detector = MLDetector()

    assert detector.model.n_features_in_ == 68
    assert len(detector.feature_names) == 68


def test_ml_prediction(sample_features):

    detector = MLDetector()

    result = detector.predict(sample_features)

    assert result["prediction"] in [
        "BENIGN",
        "DDoS"
    ]

    assert 0.0 <= result["ml_probability"] <= 1.0


def test_anomaly_detector_single_flow(sample_features):

    detector = AnomalyDetector()

    score = detector.get_anomaly_signal(
        sample_features
    )[0]

    assert 0.0 <= float(score) <= 1.0


def test_detection_pipeline(sample_features):

    pipeline = DetectionPipeline()

    result = pipeline.analyze(
        sample_features
    )

    assert result["prediction"] in [
        "BENIGN",
        "SUSPICIOUS"
    ]

    assert result["ml_prediction"] in [
        "BENIGN",
        "DDoS"
    ]

    assert 0.0 <= result["ml_probability"] <= 1.0
    assert 0.0 <= result["anomaly_score"] <= 1.0

    assert "risk_score" in result["risk"]
    assert "threat_level" in result["risk"]

    assert 0.0 <= result["risk"]["risk_score"] <= 100.0

    assert isinstance(
        result["evidence"],
        list
    )


def test_detection_pipeline_rejects_missing_features():

    pipeline = DetectionPipeline()

    with pytest.raises(ValueError):

        pipeline.analyze({
            "Destination Port": 80
        })