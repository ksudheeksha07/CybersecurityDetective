from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class AnomalyDetector:
    def __init__(self):
        base_dir = Path(__file__).resolve().parent.parent
        model_dir = base_dir / "data" / "processed"

        self.model = joblib.load(
            model_dir / "isolation_forest_benign_model.pkl"
        )

        self.imputer = joblib.load(
            model_dir / "anomaly_benign_imputer.pkl"
        )

        self.score_reference = joblib.load(
            model_dir / "anomaly_benign_score_reference.pkl"
        )

        self.feature_names = list(
            self.imputer.feature_names_in_
        )

        self.p01 = self.score_reference["p01"]
        self.p99 = self.score_reference["p99"]

    def _prepare_features(self, data):
        if isinstance(data, dict):
            dataframe = pd.DataFrame([data])

        elif isinstance(data, pd.DataFrame):
            dataframe = data.copy()

        else:
            raise TypeError(
                "Input must be a dictionary or pandas DataFrame."
            )

        dataframe.columns = (
            dataframe.columns
            .astype(str)
            .str.strip()
        )

        missing_features = [
            feature
            for feature in self.feature_names
            if feature not in dataframe.columns
        ]

        unexpected_features = [
            feature
            for feature in dataframe.columns
            if feature not in self.feature_names
        ]

        if missing_features:
            raise ValueError(
                f"Missing features: {missing_features}"
            )

        if unexpected_features:
            raise ValueError(
                f"Unexpected features: {unexpected_features}"
            )

        # Force the exact feature order used during training.
        dataframe = dataframe[self.feature_names]

        # Replace invalid numeric values.
        dataframe = dataframe.replace(
            [np.inf, -np.inf],
            np.nan
        )

        # CIC-IDS2017 uses -1 as a sentinel value
        # for some network-flow fields.
        dataframe = dataframe.replace(
            -1,
            np.nan
        )

        return dataframe

    def predict(self, data):
        dataframe = self._prepare_features(data)

        transformed = self.imputer.transform(
            dataframe
        )

        predictions = self.model.predict(
            transformed
        )

        decision_scores = self.model.decision_function(
            transformed
        )

        anomaly_scores = 1 - (
            (decision_scores - self.p01)
            / (self.p99 - self.p01)
        )

        anomaly_scores = np.clip(
            anomaly_scores,
            0,
            1
        )

        return {
            "is_anomaly": bool(
                predictions[0] == -1
            ),
            "anomaly_score": float(
                anomaly_scores[0]
            ),
            "decision_score": float(
                decision_scores[0]
            )
        }

    def get_anomaly_signal(self, data):
        dataframe = self._prepare_features(data)

        transformed = self.imputer.transform(
            dataframe
        )

        decision_scores = self.model.decision_function(
            transformed
        )

        anomaly_scores = 1 - (
            (decision_scores - self.p01)
            / (self.p99 - self.p01)
        )

        anomaly_scores = np.clip(
            anomaly_scores,
            0,
            1
        )

        return anomaly_scores