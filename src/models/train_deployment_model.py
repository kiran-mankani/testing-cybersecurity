# ============================================================
# DEPLOYMENT MODEL — STANDARD WEIGHTED XGBOOST + ISOLATION FOREST
# ============================================================
#
# Trains the model selected in reports/final_report.md for the web app:
#   * Standard Weighted XGBoost ("balanced" sample weights), same
#     hyperparameters as the fair comparison (compare_models.XGB_PARAMS);
#   * Isolation Forest fitted on normal rows only, alarm threshold at the
#     recommended 1% target false-alarm rate.
#
# Trained on ALL 132,679 cleaned rows (no hold-out): the performance
# evidence is the 5-fold CV in results/fair_comparison/, not this fit.
#
# Usage:  python -m src.models.train_deployment_model   (from the project root)

from datetime import datetime, timezone
from pathlib import Path
import json
import joblib
import numpy as np

from src.preprocessing.build_dataset import (
    CATEGORICAL_COLS,
    clean_combined,
    encode,
    TARGET_COL,
)
from src.evaluation.compare_models import XGB_PARAMS, fit_model
from src.models.anomaly import AnomalyDetector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = PROJECT_ROOT / "models" / "deployment"
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

BUNDLE_PATH = DEPLOY_DIR / "standard_weighted_xgboost_bundle.pkl"
METADATA_PATH = DEPLOY_DIR / "standard_weighted_xgboost_metadata.json"

MODEL_NAME = "Standard Weighted XGBoost"
IF_TARGET_FALSE_ALARM_RATE = 0.01


# 1. Load the corrected, deduplicated dataset
print("=" * 65)
print("DEPLOYMENT MODEL TRAINING —", MODEL_NAME.upper())
print("=" * 65)

data = clean_combined()

X_raw = data.drop(columns=[TARGET_COL])
y_raw = data[TARGET_COL].to_numpy()

class_names = np.array(sorted(np.unique(y_raw)))
y = np.searchsorted(class_names, y_raw)
normal_id = list(class_names).index("normal")

# Encoders learned from all cleaned rows (this is the training set).
X_encoded, _, encoders = encode(X_raw, X_raw)
feature_names = X_encoded.columns.tolist()

print("Training samples:", f"{len(X_encoded):,}")
print("Feature count:", len(feature_names))
print("Class count:", len(class_names))

assert X_encoded.shape[1] == 40
assert len(class_names) == 27
assert not X_encoded.isnull().any().any()

# 2. Train the classifier with the fair-comparison recipe
print("\nTraining classifier...", flush=True)
model = fit_model(name=MODEL_NAME, X=X_encoded, y=y, class_names=class_names)

assert model.n_features_in_ == len(feature_names)
assert np.array_equal(model.classes_, np.arange(len(class_names)))

# 3. Fit the Isolation Forest on normal rows only
print("Training Isolation Forest on normal rows...", flush=True)
detector = AnomalyDetector(skip_cols=CATEGORICAL_COLS).fit(X_encoded[y == normal_id])
threshold = float(detector.threshold(IF_TARGET_FALSE_ALARM_RATE))

# 4. Sanity check on the training rows (optimistic by construction)
train_pred = model.predict(X_encoded).astype(int)
train_accuracy = float((train_pred == y).mean())
print(f"\nTraining-set accuracy (not a test score): {train_accuracy:.4f}")

# 5. Save one bundle the API can load
categories = {
    col: [str(c) for c in enc.classes_] for col, enc in encoders.items()
}
metadata = {
    "model_name": MODEL_NAME,
    "anomaly_detector": "Isolation Forest (src/models/anomaly.py)",
    "if_target_false_alarm_rate": IF_TARGET_FALSE_ALARM_RATE,
    "if_threshold": threshold,
    "training_samples": int(len(X_encoded)),
    "normal_training_samples": int((y == normal_id).sum()),
    "feature_count": len(feature_names),
    "class_count": len(class_names),
    "feature_names": feature_names,
    "categorical_features": list(CATEGORICAL_COLS),
    "categories": categories,
    "classes": class_names.tolist(),
    "xgb_params": XGB_PARAMS,
    "training_method": "compute_sample_weight('balanced', y)",
    "preprocessing": (
        "src/preprocessing/build_dataset.py clean_combined() + encode(); "
        "40 features; label-encoded categoricals (unseen -> -1); no scaler."
    ),
    "training_accuracy": train_accuracy,
    "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}

bundle = {
    "model": model,
    "encoders": encoders,
    "feature_names": feature_names,
    "class_names": class_names.tolist(),
    "detector": detector,
    "metadata": metadata,
}
joblib.dump(bundle, BUNDLE_PATH)

with open(METADATA_PATH, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)

print("\nSaved bundle:", BUNDLE_PATH)
print("Saved metadata:", METADATA_PATH)
print("\nDEPLOYMENT MODEL TRAINING COMPLETED")
