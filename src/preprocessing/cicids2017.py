"""Clean CICIDS2017 (MachineLearningCVE CSVs) into one modelling table.

Source: MachineLearningCSV.zip, SHA-256
c3f26274b36c837ccf28ffd2dbf4582941c30b3ee70a635c6e5b2f87c4727928,
downloaded from the Hugging Face mirror bencorn/CICIDS2017 because the
official UNB link now redirects to a registration page. Extracted to
data/raw/cicids2017_download/MachineLearningCVE/.

Cleaning (mirrors build_dataset.py where it can):
  * strip column names, fix the mis-encoded dash in web-attack labels;
  * drop the duplicated column "Fwd Header Length.1" and constant columns;
  * drop rows with NaN / infinite values (Flow Bytes/s, Flow Packets/s);
  * remove exact duplicate rows and classes with < 10 rows;
  * keep every attack row but only a random sample of BENIGN rows, so the
    XGBoost runs fit in memory and time. Rates per class (detection, false
    alarms per benign flow) are unaffected in expectation; overall accuracy
    is not comparable to the full data.

All features are numeric, so no encoding step is needed.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_DIR = PROJECT_ROOT / "data" / "raw" / "cicids2017_download" / "MachineLearningCVE"
CACHE = PROJECT_ROOT / "data" / "processed" / "cicids2017_clean.parquet"

TARGET_COL = "Label"
NORMAL = "BENIGN"
MIN_SAMPLES = 10
BENIGN_SAMPLE = 500_000

# attack families used for the "new attack type" test
FAMILIES = {
    "DoS": ["DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest"],
    "DDoS": ["DDoS"],
    "PortScan": ["PortScan"],
    "Brute Force": ["FTP-Patator", "SSH-Patator"],
    "Web Attack": ["Web Attack - Brute Force", "Web Attack - XSS", "Web Attack - Sql Injection"],
    "Bot": ["Bot"],
    "Infiltration": ["Infiltration"],
    "Heartbleed": ["Heartbleed"],
}


def _load_raw():
    """Read each day's CSV as float32 and drop bad / duplicate rows per file
    before concatenating, to keep peak memory low."""
    frames = []
    for path in sorted(CSV_DIR.glob("*.csv")):
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)
        df.columns = df.columns.str.strip()
        df = df.drop(columns=["Fwd Header Length.1"])
        features = df.columns.drop(TARGET_COL)
        df[features] = df[features].apply(pd.to_numeric, errors="coerce").astype("float32")
        df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    # the en dash in "Web Attack – XSS" etc. is mis-encoded in the CSVs
    data[TARGET_COL] = data[TARGET_COL].str.replace(r"Web Attack \W+ ", "Web Attack - ", regex=True)
    return data


def clean(benign_sample=BENIGN_SAMPLE, seed=42):
    if CACHE.exists():
        return pd.read_parquet(CACHE)

    data = _load_raw()
    features = data.columns.drop(TARGET_COL)
    constant = [c for c in features if data[c].nunique() == 1]
    data = data.drop(columns=constant).drop_duplicates()

    counts = data[TARGET_COL].value_counts()
    data = data[data[TARGET_COL].isin(counts[counts >= MIN_SAMPLES].index)]

    benign = data[data[TARGET_COL] == NORMAL]
    benign = benign.sample(n=min(benign_sample, len(benign)), random_state=seed)
    data = pd.concat([benign, data[data[TARGET_COL] != NORMAL]])
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(CACHE, index=False)
    return data


if __name__ == "__main__":
    d = clean()
    print(d.shape)
    print(d[TARGET_COL].value_counts().to_string())
