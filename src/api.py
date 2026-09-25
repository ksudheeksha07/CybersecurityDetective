import logging
from datetime import datetime, timezone

from flask import Flask, jsonify, request

from risk_scoring import calculate_risk_score
from evidence_generator import generate_evidence
from database import (
    initialize_database,
    save_investigation,
    get_recent_investigations
)
from ml_detector import MLDetector
from detection_pipeline import DetectionPipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)

app = Flask(__name__)

initialize_database()

ml_detector = MLDetector()
detection_pipeline = DetectionPipeline()


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Cybersecurity Detective API",
        "status": "running",
        "version": "1.0"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    features = data.get("features")

    if not isinstance(features, dict):
        return jsonify({
            "error": "features must be an object"
        }), 400

    try:
        result = ml_detector.predict(features)

    except ValueError as error:
        return jsonify({
            "error": "Invalid feature data",
            "details": str(error)
        }), 400

    except Exception:
        logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed"
        }), 500

    logger.info(
        "Prediction completed: prediction=%s ml_probability=%.4f",
        result["prediction"],
        result["ml_probability"]
    )

    return jsonify({
        "prediction": result["prediction"],
        "ml_probability": result["ml_probability"]
    })


@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json()

    if not data:
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    features = data.get("features")

    if not isinstance(features, dict):
        return jsonify({
            "error": "features must be an object"
        }), 400

    logger.info(
        "Starting threat detection request: "
        "source_ip=%s destination_ip=%s protocol=%s",
        data.get("source_ip"),
        data.get("destination_ip"),
        data.get("protocol")
    )

    try:
        result = detection_pipeline.analyze(features)

    except ValueError as error:
        logger.warning(
            "Invalid feature data received: %s",
            error
        )

        return jsonify({
            "error": "Invalid feature data",
            "details": str(error)
        }), 400

    except Exception:
        logger.exception("Detection failed")

        return jsonify({
            "error": "Detection failed"
        }), 500

    logger.info(
        "Detection completed: prediction=%s ml_prediction=%s "
        "ml_probability=%.4f anomaly_score=%.4f "
        "risk_score=%.2f threat_level=%s",
        result["prediction"],
        result["ml_prediction"],
        result["ml_probability"],
        result["anomaly_score"],
        result["risk"]["risk_score"],
        result["risk"]["threat_level"]
    )

    timestamp = datetime.now(timezone.utc).isoformat()

    investigation_id = save_investigation(
        timestamp=timestamp,
        source_ip=data.get("source_ip"),
        destination_ip=data.get("destination_ip"),
        source_port=data.get("source_port"),
        destination_port=data.get("destination_port"),
        protocol=data.get("protocol"),
        prediction=result["prediction"],
        ml_probability=result["ml_probability"],
        anomaly_score=result["anomaly_score"],
        risk_score=result["risk"]["risk_score"],
        threat_level=result["risk"]["threat_level"],
        evidence=" | ".join(result["evidence"])
    )

    logger.info(
        "Investigation saved: investigation_id=%s",
        investigation_id
    )

    return jsonify({
        "investigation_id": investigation_id,
        "timestamp": timestamp,
        "prediction": result["prediction"],
        "ml_prediction": result["ml_prediction"],
        "ml_probability": result["ml_probability"],
        "anomaly_score": result["anomaly_score"],
        "rule_alerts": result["rule_alerts"],
        "risk": result["risk"],
        "evidence": result["evidence"]
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()

    if not data:
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    required_fields = [
        "ml_probability",
        "anomaly_score"
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "fields": missing_fields
        }), 400

    try:
        ml_probability = float(data["ml_probability"])
        anomaly_score = float(data["anomaly_score"])

    except (TypeError, ValueError):
        return jsonify({
            "error": "ml_probability and anomaly_score must be numeric"
        }), 400

    if not 0 <= ml_probability <= 1:
        return jsonify({
            "error": "ml_probability must be between 0 and 1"
        }), 400

    if not 0 <= anomaly_score <= 1:
        return jsonify({
            "error": "anomaly_score must be between 0 and 1"
        }), 400

    rule_alerts = data.get("rule_alerts", [])
    feature_values = data.get("features", {})

    if not isinstance(rule_alerts, list):
        return jsonify({
            "error": "rule_alerts must be a list"
        }), 400

    if not isinstance(feature_values, dict):
        return jsonify({
            "error": "features must be an object"
        }), 400

    risk = calculate_risk_score(
        ml_probability=ml_probability,
        anomaly_score=anomaly_score,
        rule_alerts=rule_alerts
    )

    evidence = generate_evidence(
        ml_probability=ml_probability,
        anomaly_score=anomaly_score,
        rule_alerts=rule_alerts,
        feature_values=feature_values
    )

    timestamp = datetime.now(timezone.utc).isoformat()

    prediction = (
        "SUSPICIOUS"
        if ml_probability >= 0.5
        else "BENIGN"
    )

    investigation_id = save_investigation(
        timestamp=timestamp,
        source_ip=data.get("source_ip"),
        destination_ip=data.get("destination_ip"),
        source_port=data.get("source_port"),
        destination_port=data.get("destination_port"),
        protocol=data.get("protocol"),
        prediction=prediction,
        ml_probability=ml_probability,
        anomaly_score=anomaly_score,
        risk_score=risk["risk_score"],
        threat_level=risk["threat_level"],
        evidence=" | ".join(evidence)
    )

    logger.info(
        "Manual analysis completed: investigation_id=%s "
        "prediction=%s risk_score=%.2f threat_level=%s",
        investigation_id,
        prediction,
        risk["risk_score"],
        risk["threat_level"]
    )

    return jsonify({
        "investigation_id": investigation_id,
        "timestamp": timestamp,
        "prediction": prediction,
        "risk": risk,
        "evidence": evidence
    })


@app.route("/investigations", methods=["GET"])
def investigations():
    limit = request.args.get(
        "limit",
        default=20,
        type=int
    )

    limit = max(1, min(limit, 100))

    results = get_recent_investigations(limit)

    investigations_list = []

    for row in results:
        investigations_list.append(dict(row))

    return jsonify({
        "count": len(investigations_list),
        "investigations": investigations_list
    })


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )