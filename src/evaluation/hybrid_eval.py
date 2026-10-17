"""Classifier + Isolation Forest on NSL-KDD.

Same two evaluations as compare_models.py, scored as attack-vs-normal:

  A. 5-fold CV on the deduplicated data (same folds and seeds), so the
     "classifier only" rows reproduce cv_summary.csv.
  B. The official split, where 1,146 test attacks belong to types never
     seen in training.

For each classifier, results are reported alone and combined with the
Isolation Forest at several target false-alarm rates. The forest is fitted
on normal training rows only; its threshold is set on held-out normal rows.

Usage:  python -m src.evaluation.hybrid_eval   (from the project root)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.compare_models import MODEL_NAMES, fit_model  # noqa: E402
from src.models.anomaly import TARGET_FALSE_ALARM_RATES, AnomalyDetector  # noqa: E402
from src.preprocessing.build_dataset import (  # noqa: E402
    CATEGORICAL_COLS,
    CONSTANT_COLS,
    TARGET_COL,
    clean_combined,
    encode,
    load_raw,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "fair_comparison"


def binary_row(model, rule, flagged, is_attack, seen=None):
    row = {
        "model": model,
        "rule": rule,
        "binary_accuracy": (flagged == is_attack).mean(),
        "detection_rate_all_attacks": flagged[is_attack].mean(),
    }
    if seen is not None:
        row["detection_rate_known_attack_types"] = flagged[is_attack & seen].mean()
        row["detection_rate_new_attack_types"] = flagged[is_attack & ~seen].mean()
    row.update({
        "false_alarm_rate": flagged[~is_attack].mean(),
        "missed_attacks": int((~flagged & is_attack).sum()),
        "false_alarms": int((flagged & ~is_attack).sum()),
    })
    return row


def score_all(clf_flags, if_flag, is_attack, seen=None):
    """Rows for IF alone, each classifier alone, and each classifier + IF.

    ``if_flag(far)`` returns the forest's alarms at target false-alarm rate ``far``.
    """
    rows = []
    for far in TARGET_FALSE_ALARM_RATES:
        rows.append(binary_row("Isolation Forest only", f"IF @ {far:.0%} target FAR",
                               if_flag(far), is_attack, seen))
    for name, flags in clf_flags.items():
        rows.append(binary_row(name, "classifier only", flags, is_attack, seen))
        for far in TARGET_FALSE_ALARM_RATES:
            rows.append(binary_row(name, f"classifier + IF @ {far:.0%} target FAR",
                                   flags | if_flag(far), is_attack, seen))
    return rows


def calibration_rank(det, X):
    """Share of the detector's held-out normal rows that score below each row of X.

    Makes scores from different folds' forests comparable: flag if rank > 1 - FAR.
    """
    return np.searchsorted(np.sort(det.calibration_scores), det.score(X)) \
        / len(det.calibration_scores)


def cross_validate(n_splits=5):
    data = clean_combined()
    class_names = np.array(sorted(data[TARGET_COL].unique()))
    y_all = np.searchsorted(class_names, data[TARGET_COL].to_numpy())
    X_all = data.drop(columns=TARGET_COL)
    normal_id = list(class_names).index("normal")

    clf_flags = {name: np.empty(len(y_all), dtype=bool) for name in MODEL_NAMES}
    if_rank = np.empty(len(y_all))
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (tr, te) in enumerate(folds.split(X_all, y_all), start=1):
        X_tr, X_te, _ = encode(X_all.iloc[tr], X_all.iloc[te])
        for name in MODEL_NAMES:
            model = fit_model(name, X_tr, y_all[tr], class_names)
            clf_flags[name][te] = model.predict(X_te).astype(int) != normal_id
        det = AnomalyDetector(skip_cols=CATEGORICAL_COLS).fit(X_tr[y_all[tr] == normal_id])
        if_rank[te] = calibration_rank(det, X_te)
        print(f"CV fold {fold} done", flush=True)

    is_attack = y_all != normal_id
    return pd.DataFrame(score_all(clf_flags, lambda far: if_rank > 1 - far, is_attack))


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

    clf_flags = {}
    for name in MODEL_NAMES:
        model = fit_model(name, X_tr, y_tr, class_names)
        clf_flags[name] = model.predict(X_te).astype(int) != normal_id
    det = AnomalyDetector(skip_cols=CATEGORICAL_COLS).fit(X_tr[y_tr == normal_id])
    rank = calibration_rank(det, X_te)
    print("official split done", flush=True)
    return pd.DataFrame(score_all(clf_flags, lambda far: rank > 1 - far, is_attack, seen))


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)

    official = official_split()
    official.to_csv(RESULTS_DIR / "hybrid_official_split.csv", index=False)
    print("\n=== Official NSL-KDD split ===")
    print(official.round(4).to_string(index=False))

    cv = cross_validate()
    cv.to_csv(RESULTS_DIR / "hybrid_cv.csv", index=False)
    print("\n=== 5-fold CV, deduplicated data ===")
    print(cv.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
