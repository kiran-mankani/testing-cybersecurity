# ============================================================
# FINAL MODEL — TRAIN CORRECTED ORIGINAL XGBOOST
# ============================================================

from pathlib import Path
import json
import joblib
import numpy as np

from src.preprocessing.build_dataset import (
    clean_combined,
    encode,
    TARGET_COL,
)
from src.evaluation.compare_models import fit_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "final_xgboost_cleaned_v2.pkl"
ENCODERS_PATH = MODEL_DIR / "final_xgboost_cleaned_v2_encoders.pkl"
FEATURES_PATH = MODEL_DIR / "final_xgboost_cleaned_v2_features.pkl"
CLASSES_PATH = MODEL_DIR / "final_xgboost_cleaned_v2_classes.pkl"
METADATA_PATH = MODEL_DIR / "final_xgboost_cleaned_v2_metadata.json"


# 1. Load the corrected, deduplicated dataset
print("=" * 65)
print("FINAL MODEL TRAINING — ORIGINAL XGBOOST")
print("=" * 65)

data = clean_combined()

X_raw = data.drop(columns=[TARGET_COL])
y_raw = data[TARGET_COL].to_numpy()

class_names = np.array(sorted(np.unique(y_raw)))
y = np.searchsorted(class_names, y_raw)

# Learn categorical encoders from the cleaned training data.
X_encoded, _, encoders = encode(X_raw, X_raw)
feature_names = X_encoded.columns.tolist()

print("Training samples:", f"{len(X_encoded):,}")
print("Feature count:", len(feature_names))
print("Class count:", len(class_names))

assert X_encoded.shape[1] == 40
assert len(class_names) == 27
assert len(y) == len(X_encoded)
assert not X_encoded.isnull().any().any()
assert np.isfinite(X_encoded.to_numpy(dtype=float)).all()

# 2. Train using the same Original XGBoost recipe used
# by the verified fair-comparison pipeline.
print("\nTraining model...", flush=True)

model = fit_model(
    name="Original XGBoost",
    X=X_encoded,
    y=y,
    class_names=class_names,
)

# 3. Verify the trained model's expected structure
assert model.n_features_in_ == len(feature_names)
assert len(model.classes_) == len(class_names)
assert np.array_equal(model.classes_, np.arange(len(class_names)))

print("\nModel training completed.")
print("Model type:", type(model).__name__)
print("Model features:", model.n_features_in_)
print("Model classes:", len(model.classes_))

# 4. Save new files without overwriting existing models
joblib.dump(model, MODEL_PATH)
joblib.dump(encoders, ENCODERS_PATH)
joblib.dump(feature_names, FEATURES_PATH)
joblib.dump(class_names.tolist(), CLASSES_PATH)

metadata = {
    "model_name": "Original XGBoost",
    "version": "cleaned_v2",
    "training_samples": int(len(X_encoded)),
    "feature_count": int(len(feature_names)),
    "class_count": int(len(class_names)),
    "feature_names": feature_names,
    "classes": class_names.tolist(),
    "training_method": "Random oversampling to configured target",
    "preprocessing": (
        "Categorical encoders learned from cleaned data; "
        "40 features; no scaler in this corrected pipeline."
    ),
}

with open(METADATA_PATH, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)

print("\nSaved model:", MODEL_PATH)
print("Saved encoders:", ENCODERS_PATH)
print("Saved features:", FEATURES_PATH)
print("Saved classes:", CLASSES_PATH)
print("Saved metadata:", METADATA_PATH)

print("\nFINAL MODEL TRAINING COMPLETED")