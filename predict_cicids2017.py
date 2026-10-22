"""Test the three XGBoost recipes on CICIDS2017 (train once, then reuse).

  1. Original XGBoost           - random oversampling of small classes
  2. Standard Weighted XGBoost  - "balanced" sample weights (selected model)
  3. Targeted Weighted XGBoost  - balanced weights x extra weight for 4 rare attacks

Same hyperparameters and recipes as src/evaluation/cicids2017_eval.py, plus
the Isolation Forest (src/models/anomaly.py) fitted on BENIGN training rows
with its alarm threshold at a 1% target false-alarm rate, as in the web app.

The models are trained on 80% of the cleaned table
(data/processed/cicids2017_clean.parquet, stratified split, seed 42) and
saved to models/cicids2017/cicids2017_xgboost_models.pkl. Every later run
loads that file instead of training again. --retrain trains and saves anew.

Two test modes:
  default     Predict the other 20% of the table, which the models never saw.
  --csv FILE  Predict FILE, any CSV with the CICFlowMeter columns. If FILE
              has a "Label" column, metrics are printed too. About 80% of
              the rows of the original MachineLearningCVE CSVs were used for
              training, so their scores are optimistic; use new traffic for
              a fair test.

For each model, a file like the web app's "Download predictions CSV" is
written: every input column plus predicted_class, confidence, verdict,
severity, response, anomaly_score and alert.

Usage (from the project root):
  .venv\\Scripts\\python.exe predict_cicids2017.py                   first run trains + saves
  .venv\\Scripts\\python.exe predict_cicids2017.py --csv flows.csv   uses the saved models
  .venv\\Scripts\\python.exe predict_cicids2017.py --retrain --sample 0.1
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost
from joblib import Parallel, delayed
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.cicids2017_eval import TARGETED_MULTIPLIERS  # noqa: E402
from src.evaluation.compare_models import MODEL_NAMES, N_PARALLEL, fit_model  # noqa: E402
from src.models.anomaly import AnomalyDetector  # noqa: E402
from src.preprocessing.cicids2017 import NORMAL, TARGET_COL, clean  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results" / "cicids2017"
MODEL_PATH = PROJECT_ROOT / "models" / "cicids2017" / "cicids2017_xgboost_models.pkl"
SMALL_CLASS = 1000  # --sample keeps every row of classes smaller than this
IF_TARGET_FALSE_ALARM_RATE = 0.01

FILE_NAMES = {
    "Original XGBoost": "original",
    "Standard Weighted XGBoost": "standard_weighted",
    "Targeted Weighted XGBoost": "targeted_weighted",
}

# Severity for CICIDS2017 classes, following the Phase 6 policy used by the
# web app (backend/model_service.py): scans Low, DoS and password guessing
# High, rootkit-like compromise Critical. Classes not listed -> "Medium".
SEVERITY_MAPPING = {
    "BENIGN": "Low",
    "PortScan": "Low",
    "Bot": "Medium",
    "Web Attack - XSS": "Medium",
    "DoS Hulk": "High", "DoS GoldenEye": "High", "DoS slowloris": "High",
    "DoS Slowhttptest": "High", "DDoS": "High",
    "FTP-Patator": "High", "SSH-Patator": "High", "Web Attack - Brute Force": "High",
    "Web Attack - Sql Injection": "High",
    "Infiltration": "Critical", "Heartbleed": "Critical",
}
RESPONSE_MAPPING = {
    "Low": "Monitor",
    "Medium": "Alert",
    "High": "Investigate",
    "Critical": "Immediate Response",
}


def sample_rows(data, frac, seed=42):
    """Keep ``frac`` of each large class and all rows of small classes, so
    rare attacks (Heartbleed has 11 rows) are not lost in a quick run."""
    parts = []
    for _, group in data.groupby(TARGET_COL):
        if len(group) > SMALL_CLASS:
            group = group.sample(n=max(SMALL_CLASS, int(len(group) * frac)), random_state=seed)
        parts.append(group)
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def load_csv(path, columns):
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df.columns = df.columns.str.strip()
    missing = [c for c in columns if c not in df.columns]
    if missing:
        sys.exit(f"{path} is missing {len(missing)} required columns, e.g. {missing[:5]}")
    df[columns] = df[columns].apply(pd.to_numeric, errors="coerce").astype("float32")
    bad = df[columns].isin([np.inf, -np.inf]).any(axis=1) | df[columns].isna().any(axis=1)
    if bad.any():
        print(f"dropping {int(bad.sum())} rows with missing or infinite values")
        df = df[~bad].reset_index(drop=True)
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].str.strip().str.replace(
            r"Web Attack \W+ ", "Web Attack - ", regex=True)
    return df


def split(data):
    """The fixed 80/20 split: the saved models are trained on the 80%."""
    labels = data[TARGET_COL].to_numpy()
    return train_test_split(np.arange(len(data)), test_size=0.2,
                            random_state=42, stratify=labels)


def train(train_rows, features, sample):
    labels = train_rows[TARGET_COL].to_numpy()
    class_names = np.array(sorted(np.unique(labels)))
    y = np.searchsorted(class_names, labels)
    X = train_rows[features]

    def run(name):
        t = time.time()
        model = fit_model(name, X, y, class_names, multipliers=TARGETED_MULTIPLIERS)
        print(f"{name}: trained in {time.time() - t:.0f}s", flush=True)
        return name, model

    print(f"training on {len(train_rows):,} rows", flush=True)
    outputs = dict(Parallel(n_jobs=N_PARALLEL, prefer="threads")(delayed(run)(n) for n in MODEL_NAMES))

    print("training Isolation Forest on BENIGN rows...", flush=True)
    detector = AnomalyDetector().fit(X[labels == NORMAL])

    bundle = {
        "models": {name: outputs[name] for name in MODEL_NAMES},
        "detector": detector,
        "if_threshold": float(detector.threshold(IF_TARGET_FALSE_ALARM_RATE)),
        "class_names": class_names,
        "features": features,
        "sample": sample,
        "train_rows": len(train_rows),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "xgboost_version": xgboost.__version__,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"saved models to {MODEL_PATH.relative_to(PROJECT_ROOT).as_posix()}", flush=True)
    return bundle


def load_bundle():
    bundle = joblib.load(MODEL_PATH)
    if "detector" not in bundle:
        sys.exit("the saved models are from an older version of this script; run with --retrain")
    print(f"loaded models from {MODEL_PATH.relative_to(PROJECT_ROOT).as_posix()} "
          f"(trained {bundle['trained_at']} on {bundle['train_rows']:,} rows, "
          f"sample {bundle['sample']})", flush=True)
    return bundle


def predict(bundle, X):
    """Per model, a table with the web app's prediction columns."""
    class_names = bundle["class_names"]
    scores = bundle["detector"].score(X)
    anomalous = scores > bundle["if_threshold"]
    results = {}
    for name, model in bundle["models"].items():
        proba = model.predict_proba(X)
        predicted = class_names[proba.argmax(axis=1)]
        is_attack = predicted != NORMAL
        # Hybrid rule from src/models/anomaly.py: classifier attack OR forest anomaly.
        verdict = np.where(is_attack, "Attack", np.where(anomalous, "Anomaly", "Normal"))
        severity = np.where(
            is_attack, [SEVERITY_MAPPING.get(c, "Medium") for c in predicted],
            np.where(anomalous, "Medium", "Low"))
        results[name] = pd.DataFrame({
            "predicted_class": predicted,
            "confidence": proba.max(axis=1),
            "verdict": verdict,
            "severity": severity,
            "response": [RESPONSE_MAPPING[s] for s in severity],
            "anomaly_score": scores,
            "alert": is_attack | anomalous,
        })
    return results


def score(results, true):
    is_attack = true != NORMAL
    classes = np.array(sorted(np.unique(true)))
    summary, per_class = [], {}
    for name, res in results.items():
        p = res["predicted_class"].to_numpy()
        flagged = p != NORMAL
        alert = res["alert"].to_numpy()
        recall = recall_score(true, p, labels=classes, average=None, zero_division=0)
        summary.append({
            "model": name,
            "accuracy": accuracy_score(true, p),
            "macro_f1": f1_score(true, p, labels=classes, average="macro", zero_division=0),
            "balanced_accuracy": recall.mean(),
            "attack_detection_rate": flagged[is_attack].mean() if is_attack.any() else np.nan,
            "false_alarm_rate": flagged[~is_attack].mean() if (~is_attack).any() else np.nan,
            "missed_attacks": int((~flagged & is_attack).sum()),
            "false_alarms": int((flagged & ~is_attack).sum()),
            "wrong_predictions": int((p != true).sum()),
            # with the Isolation Forest (the "alert" column)
            "alert_detection_rate": alert[is_attack].mean() if is_attack.any() else np.nan,
            "alert_false_alarm_rate": alert[~is_attack].mean() if (~is_attack).any() else np.nan,
        })
        per_class[name] = recall
    per_class_df = pd.DataFrame(per_class, index=classes)
    per_class_df.insert(0, "samples", pd.Series(true).value_counts().reindex(classes).to_numpy())
    return pd.DataFrame(summary), per_class_df


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, help="CICFlowMeter CSV to predict instead of the 20%% split")
    parser.add_argument("--retrain", action="store_true",
                        help="train the models again and overwrite the saved file")
    parser.add_argument("--sample", type=float, default=1.0,
                        help="when training: fraction of training rows to use, e.g. 0.1 for a quick run")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)

    data = None
    if args.retrain or not MODEL_PATH.exists():
        data = clean()
        tr, _ = split(data)
        train_rows = data.iloc[tr]
        if args.sample < 1.0:
            train_rows = sample_rows(train_rows, args.sample)
        bundle = train(train_rows, list(data.columns.drop(TARGET_COL)), args.sample)
    else:
        bundle = load_bundle()
        if args.sample < 1.0:
            print("note: --sample only applies when training; add --retrain to use it")
    features = bundle["features"]

    if args.csv:
        test = load_csv(args.csv, features)
        tag = args.csv.stem
    else:
        if data is None:
            data = clean()
        _, te = split(data)
        test = data.iloc[te].reset_index(drop=True)
        tag = "holdout"
    print(f"test rows: {len(test):,}", flush=True)

    t = time.time()
    results = predict(bundle, test[features])
    print(f"predicted in {time.time() - t:.1f}s", flush=True)

    # one file per model, like the web app's predictions CSV
    print("\nwriting prediction files...", flush=True)
    for name, res in results.items():
        pd.concat([test, res], axis=1).to_csv(
            RESULTS_DIR / f"predictions_{tag}_{FILE_NAMES[name]}.csv", index=False)

    # all three models side by side, prediction columns only
    compare = pd.DataFrame({name: res["predicted_class"] for name, res in results.items()})
    if TARGET_COL in test.columns:
        compare.insert(0, "true_label", test[TARGET_COL].to_numpy())
    compare.to_csv(RESULTS_DIR / f"predictions_{tag}.csv", index=False)

    if TARGET_COL in test.columns:
        summary, per_class = score(results, test[TARGET_COL].to_numpy())
        summary.to_csv(RESULTS_DIR / f"predict_summary_{tag}.csv", index=False)
        per_class.to_csv(RESULTS_DIR / f"predict_per_class_{tag}.csv")
        print("\n=== Overall ===")
        print(summary.round(4).to_string(index=False))
        print("\n=== Detection rate (recall) per class ===")
        print(per_class.round(3).to_string())
    else:
        print("\nno Label column: predicted class counts per model")
        print(compare.apply(pd.Series.value_counts).fillna(0).astype(int).to_string())

    print(f"\nsaved to {RESULTS_DIR.relative_to(PROJECT_ROOT).as_posix()}/:")
    for name in results:
        print(f"  predictions_{tag}_{FILE_NAMES[name]}.csv   ({name}, all columns)")
    print(f"  predictions_{tag}.csv   (3 models side by side)")
    if TARGET_COL in test.columns:
        print(f"  predict_summary_{tag}.csv, predict_per_class_{tag}.csv")


if __name__ == "__main__":
    main()
