# Final Report — Cybersecurity Model Review

## Main Model

**Standard Weighted XGBoost** has been selected as the main model.

**Reason:** Missing an attack is more dangerous than a false alarm. Compared with the Original XGBoost, Standard Weighted XGBoost misses fewer attacks (86 vs 140, −39%), but raises almost three times as many false alarms (309 vs 112; false-alarm rate 0.44% vs 0.16%). It also has higher balanced accuracy than the Original (0.886 vs 0.859), i.e. it does better on the small attack classes.

The price is slightly lower accuracy (99.62% vs 99.73%) and macro F1 (0.859 vs 0.867). The macro F1 gap is smaller than the fold-to-fold spread (std ≈ 0.013–0.015), so it is not a clear difference.

Targeted Weighted XGBoost misses 4 fewer attacks than Standard Weighted (82 vs 86) but raises 15 more false alarms and has lower macro F1; the differences are too small to justify the extra hand-tuned weights.

---

## Fair Comparison Results (Cleaned Data)

Three XGBoost models were compared with 5-fold stratified cross-validation on the deduplicated, combined NSL-KDD data (132,679 rows: 70,310 normal, 62,369 attacks, 27 classes with ≥ 10 rows). Each row is predicted once, by a model that never saw a copy of it.

| Metric | Original XGBoost | **Standard Weighted XGBoost** | Targeted Weighted XGBoost |
|--------|------------------|-------------------------------|---------------------------|
| Accuracy | 99.73% | **99.62%** | 99.61% |
| Macro F1 | 0.867 | **0.859** | 0.858 |
| Balanced accuracy | 0.859 | **0.886** | 0.886 |
| Attack detection rate | 99.78% | **99.86%** | 99.87% |
| False-alarm rate | 0.16% | **0.44%** | 0.46% |
| Missed attacks | 140 | **86** | 82 |
| False alarms | 112 | **309** | 324 |
| Wrong predictions (any class) | 355 | **504** | 515 |

The main-model column is bold to mark the selection, not because it is best on every row.

Accuracy, macro F1 and balanced accuracy are computed once over all out-of-fold predictions. The average of the five per-fold macro F1 scores is a little lower (0.858 / 0.857 / 0.853), because the rarest classes have only a few rows in each fold.

---

## Important Note

These numbers are taken from `results/fair_comparison/cv_summary.csv`, which is based on **cleaned (deduplicated) data**.

**The earlier overlapping-test-set numbers (99.75% accuracy for the Original XGBoost, 99.64% for Standard Weighted, Phase 10 step 10.5) are not used** because they were not fair: 5,086 test rows (17.1%) were copies of training rows.

---

## New Attack Results

When tested on the official NSL-KDD split (train on the training file, test on the test file), all three models catch about 93% of attacks of known types but only 24–29% of attacks of new types:

| Model | Attacks of known types caught | Attacks of new types caught | False-alarm rate |
|-------|-------------------------------|-----------------------------|------------------|
| Original XGBoost | 93.4% | 23.6% | 0.76% |
| **Standard Weighted XGBoost** | **93.4%** | **28.0%** | **0.75%** |
| Targeted Weighted XGBoost | 93.4% | 29.4% | 0.77% |

**Test set info** (`results/fair_comparison/official_split_info.json`):
- Total attacks: 11,299
- Attacks of new types (17 types never seen in training): 1,146
- Normal traffic: 11,245

This is the honest measure of how a model would cope with a brand-new attack. The main model catches only about 2 in 7 attack records of new types.

---

## Anomaly Detection (Isolation Forest)

An Isolation Forest was added next to the classifier (`src/models/anomaly.py`, evaluated by `src/evaluation/hybrid_eval.py`).

- It is trained on **normal training traffic only**, so it needs no attack labels and can react to attack types the classifier has never seen.
- Its alarm threshold is set on held-out normal rows to hit a target false-alarm rate (1%, 2% or 5%).
- **Hybrid rule:** raise an alert if the classifier predicts an attack **or** the Isolation Forest scores the record as anomalous.

### Official split — new attack types (`results/fair_comparison/hybrid_official_split.csv`)

| Main model (Standard Weighted) | Known types caught | New types caught | False-alarm rate | Missed attacks | False alarms |
|--------------------------------|--------------------|------------------|------------------|----------------|--------------|
| Classifier only | 93.4% | 28.0% | 0.75% | 1,494 | 84 |
| **+ Isolation Forest @ 1% target** | **93.5%** | **55.5%** | **1.49%** | **1,172** | **168** |
| + Isolation Forest @ 2% target | 93.5% | 62.7% | 2.34% | 1,087 | 263 |
| + Isolation Forest @ 5% target | 93.9% | 69.6% | 4.46% | 963 | 501 |
| Isolation Forest alone @ 1% target | 74.8% | 44.9% | 0.92% | 3,195 | 103 |

The other two classifiers show the same pattern (new-type detection 23.6% → 52.9% for the Original, 29.4% → 55.3% for the Targeted model, at the 1% setting).

### Cross-validation — known attack types (`results/fair_comparison/hybrid_cv.csv`)

| Main model (Standard Weighted) | Missed attacks | False alarms | False-alarm rate |
|--------------------------------|----------------|--------------|------------------|
| Classifier only | 86 | 309 | 0.44% |
| + Isolation Forest @ 1% target | 65 | 961 | 1.37% |
| + Isolation Forest @ 5% target | 59 | 3,647 | 5.19% |

**Conclusion:** The Isolation Forest roughly **doubles detection of new attack types** (28% → 56%) for about 0.75 percentage points more false alarms (84 → 168 on the official test set). On attack types the classifier already knows, it adds little: 21 fewer misses for ~650 more false alarms. The recommended setting is the **1% target**. Higher targets add false alarms faster than they add catches.

The classifier-only rows of this run reproduce `cv_summary.csv` and `official_split_binary.csv` exactly.

---

## CICIDS2017 Results

The three models were also tested on CICIDS2017 (`src/evaluation/cicids2017_eval.py`, results in `results/cicids2017/`).

**What was tested.** The saved NSL-KDD models cannot be applied directly. CICIDS2017 describes traffic with 69 different flow features, so each recipe was **retrained** on CICIDS2017 with the same settings: the same hyperparameters, with oversampling for the Original, balanced weights for Standard Weighted, and balanced weights plus extra weight for the 4 rarest attacks for Targeted Weighted (Heartbleed ×5, Infiltration ×3, SQL Injection ×2, XSS ×2).

**Data** (`src/preprocessing/cicids2017.py`): the MachineLearningCVE CSVs (2,830,743 flows). Rows with missing or infinite values and exact duplicates were removed, along with constant columns. **All 425,726 attack flows** were kept, together with a **random sample of 500,000 benign flows** to keep training time manageable (14 attack classes). Detection rates and false-alarm rates (per benign flow) are not affected by the sampling, but overall accuracy cannot be compared with the full dataset.

### Known attack types — 5-fold cross-validation (`cv_summary.csv`)

| Metric | Original XGBoost | **Standard Weighted XGBoost** | Targeted Weighted XGBoost |
|--------|------------------|-------------------------------|---------------------------|
| Accuracy | 99.84% | **99.81%** | 99.81% |
| Macro F1 | 0.901 | **0.914** | 0.912 |
| Balanced accuracy | 0.900 | **0.920** | 0.923 |
| Attack detection rate | 99.96% | **99.99%** | 99.99% |
| False-alarm rate | 0.10% | **0.14%** | 0.14% |
| Missed attacks | 175 | **46** | 45 |
| False alarms | 506 | **725** | 717 |

The NSL-KDD picture repeats. Standard Weighted misses about 4× fewer attacks than the Original for about 40% more false alarms, and here it also has the best macro F1. The weakest classes are the web attacks (`cv_per_class.csv`). The main model catches only 74% of web brute force, 62% of SQL injection and 51% of XSS, because these flows look like normal web traffic. Every other class is caught at 94% or better.

Adding the Isolation Forest does not help here. On its own, at the 1% target, it catches under 1% of attacks. Combined with the main model, it reduces misses from 46 to 42 but raises false alarms from 725 to 5,716 (`hybrid_cv.csv`).

### New attack types — leave one family out (`new_attack_families.csv`)

CICIDS2017 has no official split with unseen attack types, so one was built for this test. Each of the 8 attack families is removed from training in turn, and the model is tested on all of that family's flows, using an 80/20 stratified split for everything else. The percentages are the share of the unseen family's flows flagged as attacks:

| Unseen family (flows) | Original | **Standard Weighted** | Targeted | **Standard Weighted + IF @ 1%** |
|-----------------------|----------|-----------------------|----------|---------------------------------|
| DoS (193,730) | 2.9% | **7.0%** | 7.3% | **10.1%** |
| DDoS (128,014) | 62.9% | **16.9%** | 16.8% | **16.9%** |
| PortScan (90,694) | 1.1% | **0.7%** | 0.7% | **0.9%** |
| Brute Force (9,150) | 0.2% | **21.2%** | 20.9% | **21.2%** |
| Web Attack (2,143) | 73.1% | **59.4%** | 51.0% | **59.4%** |
| Bot (1,948) | 0.0% | **0.0%** | 0.0% | **0.0%** |
| Infiltration (36) | 0.0% | **0.0%** | 0.0% | **44.4%** |
| Heartbleed (11) | 0.0% | **0.0%** | 72.7% | **90.9%** |
| **All unseen flows (pooled)** | **20.9%** | **9.2%** | **9.2%** | **10.6%** |
| Average over families | 17.5% | 13.1% | 21.2% | 30.5% |
| False-alarm rate | 0.08% | 0.11% | 0.11% | 1.10% |

**Conclusions from CICIDS2017:**

1. **No model copes well with a new attack family.** The main model flags only 9% of flows from a family it has not seen. That is far below the 28% of new-type attacks it caught on NSL-KDD. A supervised classifier only recognises what it was trained on.
2. **Standard Weighted is not better than the Original on unseen families.** The Original catches more of the unseen flows overall (20.9% vs 9.2%), almost entirely because it catches 63% of unseen DDoS. Standard Weighted is better on unseen DoS and Brute Force. The ranking depends on which family is unseen, so this test does not favour either model. The main-model choice still rests on the known-attack results above, where Standard Weighted is clearly better.
3. **The Isolation Forest helps only for small, unusual families.** At the 1% setting it lifts Heartbleed from 0% to 91% and Infiltration from 0% to 44%. It adds almost nothing for the large DoS, DDoS, PortScan and Bot families, because those flows do not look unusual compared with benign traffic. Pooled detection of unseen flows rises only from 9.2% to 10.6%, while the false-alarm rate rises from 0.11% to 1.10%. On NSL-KDD, the same setting doubled new-attack detection.

Small families give noisy rates. Heartbleed has 11 flows and Infiltration has 36, so one flow changes Heartbleed's rate by 9 percentage points.

---

## Recommendations

1. ✅ **Keep Standard Weighted XGBoost** as the main model
2. ✅ **Drop Targeted Weighted XGBoost**
3. ✅ **Report cleaned numbers** (99.62% accuracy, 0.859 macro F1)
4. ✅ Quote the new-attack result: 24–29% of new-type attacks caught by the classifiers alone on NSL-KDD, and 9–21% of unseen-family flows on CICIDS2017
5. ✅ Add anomaly detection: **use the Isolation Forest at the 1% target false-alarm rate alongside the classifier**. It doubles new-attack detection on NSL-KDD and catches rare, unusual attacks (Heartbleed, Infiltration) on CICIDS2017. It does **not** catch new high-volume attacks (DoS, DDoS, PortScan, Bot) that resemble normal traffic, so it is a partial safety net, not a solution.
6. ✅ Test on CICIDS2017. The known-attack results confirm the choice of Standard Weighted XGBoost.
7. ⏳ Improve web-attack detection on CICIDS2017 (XSS 51%, SQL injection 62%, web brute force 74%)
8. ⏳ Try a stronger anomaly detector for new attack families (e.g. an autoencoder, or per-service baselines)
9. ⏳ Group the rarest attacks
10. ⏳ Re-run the notebooks

---

**Date:** 15 Oct 2026
**Sources:** `results/fair_comparison/cv_summary.csv`, `cv_folds.csv`, `official_split_binary.csv`, `official_split_info.json`, `hybrid_official_split.csv`, `hybrid_cv.csv`; `results/cicids2017/*.csv`
