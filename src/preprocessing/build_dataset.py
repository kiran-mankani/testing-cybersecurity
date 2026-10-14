"""Leak-free version of the Phase 2 preprocessing.

Differences from notebooks/phase2_data_preprocessing.ipynb:
  * exact duplicate rows are removed AFTER train+test are combined, so the
    same connection record can no longer appear on both sides of the split
    (in the notebook pipeline 5,086 test rows, 17.1%, were copies of
    training rows);
  * the constant column ``num_outbound_cmds`` (always 0) is dropped.

Categorical columns are label-encoded with mappings learned on the
training rows only. Scaling is not applied: the XGBoost models do not
need it.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
TARGET_COL = "labels"
CONSTANT_COLS = ["num_outbound_cmds"]
MIN_SAMPLES = 10


def load_raw():
    train = pd.read_csv(RAW_DIR / "kdd_train.csv")
    test = pd.read_csv(RAW_DIR / "kdd_test.csv")
    return train, test


def clean_combined(min_samples=MIN_SAMPLES):
    """Combine the two raw files, deduplicate and drop very rare classes."""
    train, test = load_raw()
    combined = pd.concat([train, test], ignore_index=True)
    combined = combined.drop(columns=CONSTANT_COLS)
    combined = combined.drop_duplicates().reset_index(drop=True)

    counts = combined[TARGET_COL].value_counts()
    keep = counts[counts >= min_samples].index
    return combined[combined[TARGET_COL].isin(keep)].reset_index(drop=True)


def encode(X_train, X_test):
    """Label-encode categorical columns using training categories only.

    Categories that only occur in the test rows are mapped to -1.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder().fit(X_train[col])
        mapping = {cat: i for i, cat in enumerate(le.classes_)}
        X_train[col] = X_train[col].map(mapping)
        X_test[col] = X_test[col].map(mapping).fillna(-1).astype(int)
        encoders[col] = le
    return X_train, X_test, encoders


def build(test_size=0.2, random_state=42):
    data = clean_combined()
    X = data.drop(columns=TARGET_COL)
    y = data[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_test, encoders = encode(X_train, X_test)
    return X_train, X_test, y_train, y_test, encoders


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _ = build()
    print("Train:", X_train.shape, "Test:", X_test.shape)
    print("Classes:", y_train.nunique())
