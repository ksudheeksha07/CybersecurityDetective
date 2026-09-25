import joblib
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "data" / "processed" / "random_forest_model.pkl"
PIPELINE_PATH = BASE_DIR / "data" / "processed" / "preprocessing_pipeline.pkl"


class MLDetector:

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.preprocessing_pipeline = joblib.load(PIPELINE_PATH)

        # Feature schema used during model training
        self.feature_names = list(
            self.preprocessing_pipeline.feature_names_in_
        )

    def validate_features(self, features):

        if not isinstance(features, dict):
            raise ValueError("Features must be provided as an object.")

        received_features = set(features.keys())
        expected_features = set(self.feature_names)

        missing_features = expected_features - received_features
        unexpected_features = received_features - expected_features

        if missing_features:
            raise ValueError(
                f"Missing features: {sorted(missing_features)}"
            )

        if unexpected_features:
            raise ValueError(
                f"Unexpected features: {sorted(unexpected_features)}"
            )

        if len(features) != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, "
                f"received {len(features)}."
            )

    def predict(self, features):

        self.validate_features(features)

        # Create dataframe using the exact training feature order
        dataframe = pd.DataFrame(
            [[features[name] for name in self.feature_names]],
            columns=self.feature_names
        )

        processed_features = self.preprocessing_pipeline.transform(
            dataframe
        )

        prediction = self.model.predict(
            processed_features
        )[0]

        probabilities = self.model.predict_proba(
            processed_features
        )[0]

        class_probabilities = dict(
            zip(
                self.model.classes_,
                probabilities
            )
        )

        ddos_probability = float(
            class_probabilities.get("DDoS", 0.0)
        )

        return {
            "prediction": str(prediction),
            "ml_probability": ddos_probability
        }


if __name__ == "__main__":

    detector = MLDetector()

    print("ML detector loaded successfully.")
    print(f"Model: {type(detector.model).__name__}")
    print(f"Features expected: {len(detector.feature_names)}")
    print("Feature validation: enabled")