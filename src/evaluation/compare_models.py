"""Fair comparison of the three XGBoost variants from Phases 4, 9 and 10.

  1. Original XGBoost        - random oversampling of small classes to 1,000 rows (Phase 3/4)
  2. Standard Weighted XGBoost - "balanced" sample weights (Phase 9)
  3. Targeted Weighted XGBoost - balanced weights x extra multipliers for 4 rare attacks (Phase 10)

All three use the same hyperparameters as the saved models. Two evaluations:

  A. 5-fold stratified cross-validation on the deduplicated, combined data
     (src/preprocessing/build_dataset.py). Every row is tested exactly once
     and never by a model that saw an identical copy of it.
  B. The official NSL-KDD split: train on kdd_train.csv only, test on
     kdd_test.csv, which contains 17 attack types the model has never seen.
     Scored as attack-vs-normal, since unseen types cannot be named.

Usage:  python -m src.evaluation.compare_models   (from the project root)
"""
import json
import sys
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
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.build_dataset import (  # noqa: E402
    CONSTANT_COLS,
    TARGET_COL,
    clean_combined,
    encode,
    load_raw,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "fair_comparison"

XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=3,
)
N_PARALLEL = 3  # models trained at the same time (3 x 3 threads)

OVERSAMPLE_TO = 1000
TARGETED_MULTIPLIERS = {
    "rootkit": 5.0,
    "multihop": 3.0,
    "ftp_write": 2.0,
    "loadmodule": 2.0,
}
MODEL_NAMES = [
    "Original XGBoost",
    "Standard Weighted XGBoost",
    "Targeted Weighted XGBoost",
]


def oversample(X, y, target=OVERSAMPLE_TO, seed=42):
    """Randomly duplicate rows of classes smaller than ``target``."""
    rng = np.random.default_rng(seed)
    idx = [np.arange(len(y))]
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        if len(cls_idx) < target:
            idx.append(rng.choice(cls_idx, target - len(cls_idx), replace=True))
    idx = np.concatenate(idx)
    return X.iloc[idx], y[idx]


def fit_model(name, X, y, class_names, multipliers=TARGETED_MULTIPLIERS):
    model = XGBClassifier(num_class=len(class_names), **XGB_PARAMS)
    if name == "Original XGBoost":
        X_os, y_os = oversample(X, y)
        return model.fit(X_os, y_os)

    weights = compute_sample_weight("balanced", y)
    if name == "Targeted Weighted XGBoost":
        for attack, mult in multipliers.items():
            # ftp_write has only 9 unique rows, so deduplication drops it
            if attack in class_names:
                weights[y == list(class_names).index(attack)] *= mult
    return model.fit(X, y, sample_weight=weights)


def cross_validate(n_splits=5):
    data = clean_combined()
    class_names = np.array(sorted(data[TARGET_COL].unique()))
    y_all = np.searchsorted(class_names, data[TARGET_COL].to_numpy())
    X_all = data.drop(columns=TARGET_COL)
    normal_id = list(class_names).index("normal")

    preds = {name: np.empty(len(y_all), dtype=int) for name in MODEL_NAMES}
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(folds.split(X_all, y_all))

    def run(fold, name):
        tr, te = splits[fold - 1]
        X_tr, X_te, _ = encode(X_all.iloc[tr], X_all.iloc[te])
        model = fit_model(name, X_tr, y_all[tr], class_names)
        p = model.predict(X_te).astype(int)
        print(f"fold {fold} {name}: macro F1 "
              f"{f1_score(y_all[te], p, average='macro', zero_division=0):.4f}", flush=True)
        return fold, name, p

    tasks = [(f, n) for f in range(1, n_splits + 1) for n in MODEL_NAMES]
    outputs = Parallel(n_jobs=N_PARALLEL, prefer="threads")(delayed(run)(f, n) for f, n in tasks)

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

    folds_df = pd.DataFrame(fold_rows)
    summary, per_class = [], {}
    for name in MODEL_NAMES:
        p = preds[name]
        is_attack, flagged = y_all != normal_id, p != normal_id
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
    return pd.DataFrame(summary), folds_df, per_class_df


def official_split():
    train, test = load_raw()
    train = train.drop(columns=CONSTANT_COLS)
    test = test.drop(columns=CONSTANT_COLS)
    class_names = np.array(sorted(train[TARGET_COL].unique()))
    normal_id = list(class_names).index("normal")
    y_tr = np.searchsorted(class_names, train[TARGET_COL].to_numpy())
    X_tr, X_te, _ = encode(train.drop(columns=TARGET_COL), test.drop(columns=TARGET_COL))

    test_labels = test[TARGET_COL].to_numpy()
    is_attack = test_labels != "normal"
    seen = np.isin(test_labels, class_names)

    fitted = Parallel(n_jobs=N_PARALLEL, prefer="threads")(
        delayed(fit_model)(name, X_tr, y_tr, class_names) for name in MODEL_NAMES
    )
    rows = []
    for name, model in zip(MODEL_NAMES, fitted):
        flagged = model.predict(X_te).astype(int) != normal_id
        rows.append({
            "model": name,
            "binary_accuracy": (flagged == is_attack).mean(),
            "detection_rate_all_attacks": flagged[is_attack].mean(),
            "detection_rate_known_attack_types": flagged[is_attack & seen].mean(),
            "detection_rate_new_attack_types": flagged[is_attack & ~seen].mean(),
            "false_alarm_rate": flagged[~is_attack].mean(),
        })
        print(f"official split {name}: detection {rows[-1]['detection_rate_all_attacks']:.4f}", flush=True)
    info = {
        "test_attacks": int(is_attack.sum()),
        "test_attacks_of_new_types": int((is_attack & ~seen).sum()),
        "test_normal": int((~is_attack).sum()),
    }
    return pd.DataFrame(rows), info


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary, folds_df, per_class = cross_validate()
    summary.to_csv(RESULTS_DIR / "cv_summary.csv", index=False)
    folds_df.to_csv(RESULTS_DIR / "cv_folds.csv", index=False)
    per_class.to_csv(RESULTS_DIR / "cv_per_class.csv")

    official, info = official_split()
    official.to_csv(RESULTS_DIR / "official_split_binary.csv", index=False)
    (RESULTS_DIR / "official_split_info.json").write_text(json.dumps(info, indent=2))

    pd.set_option("display.width", 200)
    print("\n=== 5-fold CV, deduplicated data ===")
    print(summary.round(4).to_string(index=False))
    print("\n=== Official NSL-KDD split (attack vs normal) ===", info)
    print(official.round(4).to_string(index=False))
    print("\n=== Per-class recall, classes with < 100 samples ===")
    print(per_class[per_class.samples < 100].round(3).to_string())


if __name__ == "__main__":
    main()
