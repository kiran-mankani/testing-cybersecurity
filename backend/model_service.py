"""Loads the deployment bundle and turns raw NSL-KDD records into predictions.

The bundle is written by src/models/train_deployment_model.py. Preprocessing
mirrors src/preprocessing/build_dataset.encode(): label-encode the three
categorical columns with the training categories, unseen categories -> -1.
"""
import time

import joblib
import numpy as np
import pandas as pd

from backend import config

# Phase 6 severity / response policy (notebooks/phase6_threat_response.ipynb,
# SEVERITY_MAPPING and RESPONSE_MAPPING). Classes not listed default to
# "Medium", as in the notebook's get_severity().
SEVERITY_MAPPING = {
    "normal": "Low",
    "ipsweep": "Low", "portsweep": "Low", "nmap": "Low",
    "satan": "Medium", "warezclient": "Medium", "warezmaster": "Medium",
    "imap": "Medium", "multihop": "Medium", "loadmodule": "Medium",
    "neptune": "High", "back": "High", "land": "High", "pod": "High",
    "teardrop": "High", "smurf": "High", "guess_passwd": "High",
    "buffer_overflow": "High",
    "rootkit": "Critical",
}
RESPONSE_MAPPING = {
    "Low": "Monitor",
    "Medium": "Alert",
    "High": "Investigate",
    "Critical": "Immediate Response",
}
TOP_K = 5


class ModelNotLoaded(RuntimeError):
    pass


class ModelService:
    def __init__(self, bundle_path=config.MODEL_BUNDLE_PATH):
        self.bundle_path = bundle_path
        self.bundle = None
        self.error = None

    def load(self):
        try:
            self.bundle = joblib.load(self.bundle_path)
            self.error = None
        except FileNotFoundError:
            self.error = (f"Model bundle not found at {self.bundle_path}. "
                          "Run: python -m src.models.train_deployment_model")
        return self

    @property
    def ready(self):
        return self.bundle is not None

    @property
    def metadata(self):
        if not self.ready:
            raise ModelNotLoaded(self.error)
        return self.bundle["metadata"]

    def _encode(self, records):
        """records: list of dicts with the 40 feature names -> (matrix, warnings per row)."""
        X = pd.DataFrame.from_records(records)[self.bundle["feature_names"]]
        warnings = [[] for _ in range(len(X))]
        for col, encoder in self.bundle["encoders"].items():
            mapping = {cat: i for i, cat in enumerate(encoder.classes_)}
            codes = X[col].map(mapping)
            for i in np.flatnonzero(codes.isna().to_numpy()):
                warnings[i].append(f"{col}='{X[col].iat[i]}' was not seen in training; encoded as -1")
            X[col] = codes.fillna(-1).astype(int)
        return X.astype(float), warnings

    def predict(self, records):
        if not self.ready:
            raise ModelNotLoaded(self.error)
        start = time.perf_counter()
        X, warnings = self._encode(records)
        classes = np.array(self.bundle["class_names"])
        proba = self.bundle["model"].predict_proba(X)
        detector = self.bundle["detector"]
        meta = self.bundle["metadata"]
        scores = detector.score(X)
        anomalous = scores > meta["if_threshold"]
        elapsed_ms = (time.perf_counter() - start) * 1000

        results = []
        for i, row in enumerate(proba):
            order = np.argsort(row)[::-1]
            predicted = str(classes[order[0]])
            is_attack = predicted != "normal"
            # Hybrid rule from src/models/anomaly.py: classifier attack OR forest anomaly.
            if is_attack:
                severity = SEVERITY_MAPPING.get(predicted, "Medium")
                verdict = "Attack"
            elif anomalous[i]:
                severity = "Medium"
                verdict = "Anomaly"
            else:
                severity = "Low"
                verdict = "Normal"
            results.append({
                "predicted_class": predicted,
                "confidence": float(row[order[0]]),
                "top_classes": [
                    {"class": str(classes[j]), "probability": float(row[j])} for j in order[:TOP_K]
                ],
                "classifier_says_attack": bool(is_attack),
                "anomaly_score": float(scores[i]),
                "anomaly_threshold": float(meta["if_threshold"]),
                "anomaly_flag": bool(anomalous[i]),
                "alert": bool(is_attack or anomalous[i]),
                "verdict": verdict,
                "severity": severity,
                "response": RESPONSE_MAPPING[severity],
                "warnings": warnings[i],
            })
        return results, elapsed_ms
