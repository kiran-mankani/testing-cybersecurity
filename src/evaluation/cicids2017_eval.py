"""The three XGBoost recipes (+ Isolation Forest) on CICIDS2017.

The saved NSL-KDD models cannot be applied here: CICIDS2017 has 69 flow
features (CICFlowMeter), not NSL-KDD's 40. Each recipe is retrained with the
same hyperparameters, oversampling / balanced weights / targeted weights.
Targeted weights go to the four rarest attacks, mirroring the 5/3/2/2
multipliers used on NSL-KDD.

  A. 5-fold stratified CV on src/preprocessing/cicids2017.py (multi-class),
     plus attack-vs-benign results with the Isolation Forest added.
  B. New attack types: CICIDS2017 has no official split with unseen
     attacks, so each attack family in turn is removed from training
     (80/20 stratified split) and the test set gets ALL of that family's
     rows. Reports how often the unseen family is flagged.

Usage:  python -m src.evaluation.cicids2017_eval   (from the project root)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.compare_models import MODEL_NAMES, N_PARALLEL, fit_model  # noqa: E402
from src.evaluation.hybrid_eval import binary_row, calibration_rank, score_all  # noqa: E402
from src.models.anomaly import TARGET_FALSE_ALARM_RATES, AnomalyDetector  # noqa: E402
from src.preprocessing.cicids2017 import FAMILIES, NORMAL, TARGET_COL, clean  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results" / "cicids2017"

TARGETED_MULTIPLIERS = {
    "Heartbleed": 5.0,
    "Infiltration": 3.0,
    "Web Attack - Sql Injection": 2.0,
    "Web Attack - XSS": 2.0,
}


def _fit(name, X, labels):
    class_names = np.array(sorted(np.unique(labels)))
    y = np.searchsorted(class_names, labels)
    model = fit_model(name, X, y, class_names, multipliers=TARGETED_MULTIPLIERS)
    return model, class_names


def cross_validate(data, n_splits=5):
    labels = data[TARGET_COL].to_numpy()
    class_names = np.array(sorted(np.unique(labels)))
    y_all = np.searchsorted(class_names, labels)
    X_all = data.drop(columns=TARGET_COL)
    normal_id = list(class_names).index(NORMAL)

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(folds.split(X_all, y_all))

    def run(fold, name):
        tr, te = splits[fold - 1]
        t = time.time()
        model, _ = _fit(name, X_all.iloc[tr], labels[tr])
        p = model.predict(X_all.iloc[te]).astype(int)
        print(f"CV fold {fold} {name}: macro F1 "
              f"{f1_score(y_all[te], p, average='macro', zero_division=0):.4f} "
              f"({time.time() - t:.0f}s)", flush=True)
        return fold, name, p

    tasks = [(f, n) for f in range(1, n_splits + 1) for n in MODEL_NAMES]
    outputs = Parallel(n_jobs=N_PARALLEL, prefer="threads")(delayed(run)(f, n) for f, n in tasks)

    preds = {name: np.empty(len(y_all), dtype=int) for name in MODEL_NAMES}
    fold_rows = []
    for fold, name, p in outputs:
        te = splits[fold - 1][1]
        preds[name][te] = p
        fold_rows.append({
            "fold": fold,
            "model": name,
            "accuracy": accuracy_score(y_all[te], p),
            "macro_f1": f1_score(y_all[te], p, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_all[te], p),
        })
    folds_df = pd.DataFrame(fold_rows).sort_values(["fold", "model"])

    if_rank = np.empty(len(y_all))
    for fold, (tr, te) in enumerate(splits, start=1):
        det = AnomalyDetector().fit(X_all.iloc[tr][y_all[tr] == normal_id])
        if_rank[te] = calibration_rank(det, X_all.iloc[te])

    is_attack = y_all != normal_id
    summary, per_class = [], {}
    for name in MODEL_NAMES:
        p = preds[name]
        flagged = p != normal_id
        f = folds_df[folds_df.model == name]
        summary.append({
            "model": name,
            "accuracy": accuracy_score(y_all, p),
            "macro_f1": f1_score(y_all, p, average="macro", zero_division=0),
            "macro_f1_fold_std": f["macro_f1"].std(),
            "balanced_accuracy": balanced_accuracy_score(y_all, p),
            "attack_detection_rate": flagged[is_attack].mean(),
            "false_alarm_rate": flagged[~is_attack].mean(),
            "missed_attacks": int((~flagged & is_attack).sum()),
            "false_alarms": int((flagged & ~is_attack).sum()),
            "wrong_predictions": int((p != y_all).sum()),
        })
        per_class[name] = recall_score(
            y_all, p, labels=range(len(class_names)), average=None, zero_division=0
        )
        per_class[name + " F1"] = f1_score(
            y_all, p, labels=range(len(class_names)), average=None, zero_division=0
        )
    per_class_df = pd.DataFrame(per_class, index=class_names)
    per_class_df.insert(0, "samples", np.bincount(y_all))

    clf_flags = {name: preds[name] != normal_id for name in MODEL_NAMES}
    hybrid = pd.DataFrame(score_all(clf_flags, lambda far: if_rank > 1 - far, is_attack))
    return pd.DataFrame(summary), folds_df, per_class_df, hybrid


def new_attack_test(data):
    """Leave one attack family out of training; test on all of its rows."""
    labels = data[TARGET_COL].to_numpy()
    X_all = data.drop(columns=TARGET_COL)
    tr, te = train_test_split(np.arange(len(data)), test_size=0.2,
                              random_state=42, stratify=labels)

    # benign training rows don't depend on the held-out family: one forest
    det = AnomalyDetector().fit(X_all.iloc[tr][labels[tr] == NORMAL])

    def run(family, name):
        members = np.isin(labels, FAMILIES[family])
        train_idx = tr[~members[tr]]
        test_idx = np.concatenate([te[~members[te]], np.flatnonzero(members)])
        t = time.time()
        model, class_names = _fit(name, X_all.iloc[train_idx], labels[train_idx])
        p = class_names[model.predict(X_all.iloc[test_idx]).astype(int)]
        print(f"new-attack {family} / {name} ({time.time() - t:.0f}s)", flush=True)
        return family, name, test_idx, p != NORMAL

    tasks = [(fam, n) for fam in FAMILIES for n in MODEL_NAMES]
    outputs = Parallel(n_jobs=N_PARALLEL, prefer="threads")(delayed(run)(f, n) for f, n in tasks)

    rows = []
    for family, name, test_idx, flagged in outputs:
        test_labels = labels[test_idx]
        is_attack = test_labels != NORMAL
        new = np.isin(test_labels, FAMILIES[family])
        if_rank = calibration_rank(det, X_all.iloc[test_idx])
        rules = [("classifier only", flagged)] + [
            (f"classifier + IF @ {far:.0%} target FAR", flagged | (if_rank > 1 - far))
            for far in TARGET_FALSE_ALARM_RATES
        ]
        for rule, f in rules:
            row = binary_row(name, rule, f, is_attack, seen=~new)
            rows.append({"held_out_family": family, "held_out_rows": int(new.sum()), **row})
    return pd.DataFrame(rows)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    data = clean()
    print("data", data.shape, flush=True)

    summary, folds_df, per_class, hybrid = cross_validate(data)
    summary.to_csv(RESULTS_DIR / "cv_summary.csv", index=False)
    folds_df.to_csv(RESULTS_DIR / "cv_folds.csv", index=False)
    per_class.to_csv(RESULTS_DIR / "cv_per_class.csv")
    hybrid.to_csv(RESULTS_DIR / "hybrid_cv.csv", index=False)
    print("\n=== CICIDS2017 5-fold CV ===")
    print(summary.round(4).to_string(index=False))
    print(per_class.round(3).to_string())
    print(hybrid.round(4).to_string(index=False))

    new = new_attack_test(data)
    new.to_csv(RESULTS_DIR / "new_attack_families.csv", index=False)
    print("\n=== CICIDS2017 leave-one-family-out ===")
    print(new.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
