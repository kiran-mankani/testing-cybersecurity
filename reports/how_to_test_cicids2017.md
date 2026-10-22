# How to Test the Models on CICIDS2017

## Why the website cannot be used

The Streamlit frontend and the FastAPI backend accept the **40 NSL-KDD features**
(`duration`, `protocol_type`, `service`, `src_bytes`, ...), and they serve the
NSL-KDD model `models/deployment/standard_weighted_xgboost_bundle.pkl`.

CICIDS2017 uses **69 different flow features** produced by CICFlowMeter
(`Flow Duration`, `Total Fwd Packets`, `Flow Bytes/s`, ...). None of the
NSL-KDD fields exist in CICIDS2017, so the web form cannot take a CICIDS2017
flow and the deployed model cannot score one.

CICIDS2017 testing is therefore done **from the command line**. The models
are retrained on CICIDS2017 with the same settings and evaluated there.
`predict_cicids2017.py` (step 6) saves its trained CICIDS2017 models and
reuses them; the full evaluation in step 4 saves only result tables.

---

## 1. Prerequisites

- The project virtual environment (`.venv`) with the packages already
  installed (pandas, numpy, scikit-learn, xgboost, joblib, pyarrow).
- About **8 GB of free RAM** and a few GB of disk space.
- Run every command from the project root:
  `D:\Vinita-Chawla\Research-projects\testing-cybersecurity`

## 2. Get the dataset

The data is already downloaded in this project. If it ever has to be
fetched again:

1. Download `MachineLearningCSV.zip` (the official UNB link now goes to a
   registration page; the copy used here came from the Hugging Face mirror
   `bencorn/CICIDS2017`).
   SHA-256: `c3f26274b36c837ccf28ffd2dbf4582941c30b3ee70a635c6e5b2f87c4727928`
2. Extract it so the 8 CSV files sit in:

```
data/raw/cicids2017_download/MachineLearningCVE/
    Monday-WorkingHours.pcap_ISCX.csv
    Tuesday-WorkingHours.pcap_ISCX.csv
    Wednesday-workingHours.pcap_ISCX.csv
    Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
    Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
    Friday-WorkingHours-Morning.pcap_ISCX.csv
    Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
    Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

## 3. Clean the data

```powershell
.venv\Scripts\python.exe -m src.preprocessing.cicids2017
```

This uses `src/preprocessing/cicids2017.py` to:

- strip column names and fix the mis-encoded dash in the web-attack labels;
- drop the duplicated column `Fwd Header Length.1` and constant columns;
- drop rows with missing or infinite values, and exact duplicate rows;
- drop classes with fewer than 10 rows;
- keep **all attack flows** but only a **random sample of 500,000 benign
  flows**, so training fits in memory and time.

The result is cached in `data/processed/cicids2017_clean.parquet`
(already present). It prints the table shape and the row count per class.
**To rebuild from the CSVs, delete the parquet file first**, otherwise the
cache is reused.

## 4. Run the full evaluation

```powershell
.venv\Scripts\python.exe -m src.evaluation.cicids2017_eval
```

`src/evaluation/cicids2017_eval.py` retrains the three XGBoost recipes plus
the Isolation Forest and runs two tests:

| Test | What it does |
|------|--------------|
| **A. Known attacks (5-fold CV)** | Stratified 5-fold cross-validation on the cleaned table. Every model is trained on 4 folds and tested on the fifth. |
| **B. New attack types** | Each of the 8 attack families (DoS, DDoS, PortScan, Brute Force, Web Attack, Bot, Infiltration, Heartbleed) is removed from training in turn. The model is tested on all of that family's flows to see how many it still flags. |

It trains about 40 XGBoost models, so expect it to take **a long time
(tens of minutes to a few hours, depending on the CPU)**. Progress is printed
per fold and per family.

## 5. Read the results

The tables are written to `results/cicids2017/`:

| File | Contents |
|------|----------|
| `cv_summary.csv` | One row per model: accuracy, macro F1, balanced accuracy, attack detection rate, false-alarm rate, missed attacks, false alarms |
| `cv_folds.csv` | The same metrics for each of the 5 folds |
| `cv_per_class.csv` | Detection rate (recall) and F1 for each attack class |
| `hybrid_cv.csv` | Classifier + Isolation Forest at 1%, 2% and 5% target false-alarm rates |
| `new_attack_families.csv` | Test B: share of each unseen family's flows that are flagged, with and without the Isolation Forest |

Results from the last run (Standard Weighted XGBoost, the selected model):

- Macro F1 **0.914**, attack detection rate **99.99%**, false-alarm rate **0.14%**.
- Weakest classes: XSS 51%, SQL injection 62%, web brute force 74%.
- Unseen attack families: only **9.2%** of their flows are flagged
  (10.6% with the Isolation Forest at 1%).

A full discussion is in `reports/final_report.md`, section *CICIDS2017 Results*.

---

## 6. Test all 3 models with `predict_cicids2017.py`

`predict_cicids2017.py` (project root) is a quicker test than step 4. It
tests Original, Standard Weighted and Targeted Weighted XGBoost side by side.

**Train once, reuse afterwards.** The models are always trained on the same
80% of the cleaned table (stratified split, seed 42). The **first run** trains
all three and saves them to:

```
models/cicids2017/cicids2017_xgboost_models.pkl
```

**Every later run loads this file and only predicts**, which takes seconds.
The file also stores the Isolation Forest and its threshold, the class names,
the feature list, the training date,
the number of training rows and the XGBoost version, and these are printed
when it loads. It is ignored by git, like the other model files.

**a) Held-out test (recommended).** The models predict the other 20% of the
table, which they never saw during training:

```powershell
.venv\Scripts\python.exe predict_cicids2017.py
```

**b) Your own CSV file.** The saved models predict the given file. The file
needs the CICFlowMeter columns. If it has a `Label` column, metrics are
printed; if not, only the predicted class counts are printed:

```powershell
.venv\Scripts\python.exe predict_cicids2017.py --csv path\to\flows.csv
```

**c) Train again.** `--retrain` trains the models again and overwrites the
saved file. Add `--sample 0.1` to train on 10% of each large class (all rows
of classes under 1,000 are kept), which takes a few minutes:

```powershell
.venv\Scripts\python.exe predict_cicids2017.py --retrain
.venv\Scripts\python.exe predict_cicids2017.py --retrain --sample 0.1
```

`--sample` only matters when training: either on the first run, or together
with `--retrain`. If you train with `--sample`, that smaller model is the one
saved and reused until you run `--retrain` again.

**Output** (printed and saved in `results/cicids2017/`, where `<tag>` is
`holdout` or the CSV file name):

| File | Contents |
|------|----------|
| `predictions_<tag>_original.csv` | Original XGBoost: every input column + the prediction columns below |
| `predictions_<tag>_standard_weighted.csv` | Standard Weighted XGBoost: same layout |
| `predictions_<tag>_targeted_weighted.csv` | Targeted Weighted XGBoost: same layout |
| `predictions_<tag>.csv` | The 3 models side by side: true label (if known) and each model's predicted class |
| `predict_summary_<tag>.csv` | Per model: accuracy, macro F1, balanced accuracy, attack detection rate, false-alarm rate, missed attacks, false alarms, and the detection / false-alarm rates of the `alert` column |
| `predict_per_class_<tag>.csv` | Detection rate (recall) per class for each model |

The three per-model files have the same layout as the website's **Download
predictions CSV**. The input columns (including `Label`, if present) come
first, then:

| Column | Meaning |
|--------|---------|
| `predicted_class` | Class the model predicts (`BENIGN`, `DDoS`, `PortScan`, ...) |
| `confidence` | The model's probability for that class (0–1) |
| `verdict` | `Attack` = model predicts an attack class; `Anomaly` = model says BENIGN but the Isolation Forest finds the flow unusual; `Normal` = neither |
| `severity` | `Low` / `Medium` / `High` / `Critical` (table below); an `Anomaly` is `Medium` |
| `response` | Low → Monitor, Medium → Alert, High → Investigate, Critical → Immediate Response |
| `anomaly_score` | Isolation Forest score; higher = more unusual. Above the threshold (set so about 1% of benign flows exceed it) the flow is anomalous |
| `alert` | `True` if the verdict is `Attack` or `Anomaly` |

The Isolation Forest is trained on the BENIGN training flows and saved with
the models; it is the same for all three files.

Severity of each CICIDS2017 class, following the website's NSL-KDD policy
(scans Low, DoS and password guessing High, system compromise Critical):

| Severity | Classes |
|----------|---------|
| Low | BENIGN, PortScan |
| Medium | Bot, Web Attack - XSS |
| High | DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest, DDoS, FTP-Patator, SSH-Patator, Web Attack - Brute Force, Web Attack - Sql Injection |
| Critical | Infiltration, Heartbleed |

For the held-out test (about 185,000 flows), each per-model file is about
90 MB.

Notes:

- About 80% of the rows of the original `MachineLearningCVE/` CSVs were used
  for training. If you pass one of them with `--csv`, the scores are too
  high. For a fair test, use mode (a) or a CSV of new traffic exported with
  CICFlowMeter.
- Rows with missing or infinite values are dropped, and the script reports
  how many.















simple:
you gave me code to train the model on cicids dataset on all 3 models and then predict file use that model to predict the cicids csv files and give me updatd  csv file with predicted, verdict and all columns. so now i can test all csv files one by one and get the updated csv file?
1. It's one file, not two. predict_cicids2017.py does both jobs. The first time you run it, it trains all 3 models and saves them to models/cicids2017/cicids2017_xgboost_models.pkl. Every run after that just loads them and predicts. You don't run a separate training script.
2. Yes, you can test the CSV files one by one:
.venv\Scripts\python.exe predict_cicids2017.py --csv "data\raw\cicids2017_download\MachineLearningCVE\Tuesday-WorkingHours.pcap_ISCX.csv"
   If the models haven't been trained yet, that first --csv run trains them before predicting, so it will be slow. Every run after that is fast.
3. Each CSV gives you 3 updated files, not 1. There's one per model, in results/cicids2017/. For the Tuesday file, for example:
   - predictions_Tuesday-WorkingHours.pcap_ISCX_original.csv
   - predictions_Tuesday-WorkingHours.pcap_ISCX_standard_weighted.csv
   - predictions_Tuesday-WorkingHours.pcap_ISCX_targeted_weighted.csv

   Each has all the original columns plus predicted_class, confidence, verdict, severity, response, anomaly_score and alert. You also get a side-by-side comparison file and summary files for that CSV.
4. Results on these 8 CSVs will look too good. About 80% of their rows were used to train the models. They're fine for seeing how the output looks, but they aren't a fair accuracy test. For fair numbers, run the script with no --csv; it tests only on the 20% the models never saw. You could also use new traffic captured with CICFlowMeter.
5. Some output files will be big. The output files are roughly as large as the input CSV. Monday's file has about 530,000 rows, so each of its 3 output files will be a few hundred MB.