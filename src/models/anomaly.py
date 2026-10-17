"""Isolation Forest anomaly detector used alongside the XGBoost classifier.

The forest is trained on NORMAL traffic only, so it needs no attack labels
and can flag attack types the classifier has never seen. Its alarm threshold
is set on held-out normal rows so that a chosen share of normal traffic
(the target false-alarm rate) is flagged.

Hybrid rule: raise an alert if the classifier predicts any attack class OR
the forest scores the record above its threshold.
"""
import numpy as np
from sklearn.ensemble import IsolationForest

IF_PARAMS = dict(n_estimators=200, max_samples=256, random_state=42, n_jobs=3)
TARGET_FALSE_ALARM_RATES = (0.01, 0.02, 0.05)


def _transform(X, skip_cols=()):
    """log1p numeric columns so heavy-tailed counts/bytes don't dominate splits."""
    X = X.astype(float).copy()
    for col in X.columns:
        if col not in skip_cols:
            X[col] = np.log1p(X[col].clip(lower=0))
    return X.to_numpy()


class AnomalyDetector:
    def __init__(self, skip_cols=(), calibration_share=0.2, seed=42):
        self.skip_cols = tuple(skip_cols)
        self.calibration_share = calibration_share
        self.seed = seed

    def fit(self, X_normal):
        """Fit on normal rows; keep a held-out slice to calibrate thresholds."""
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(X_normal))
        n_cal = int(len(order) * self.calibration_share)
        cal, fit = order[:n_cal], order[n_cal:]
        self.forest = IsolationForest(**IF_PARAMS).fit(
            _transform(X_normal.iloc[fit], self.skip_cols)
        )
        self.calibration_scores = self.score(X_normal.iloc[cal])
        return self

    def score(self, X):
        """Higher = more anomalous."""
        return -self.forest.score_samples(_transform(X, self.skip_cols))

    def threshold(self, false_alarm_rate):
        return np.quantile(self.calibration_scores, 1 - false_alarm_rate)

    def flag(self, X, false_alarm_rate):
        return self.score(X) > self.threshold(false_alarm_rate)
