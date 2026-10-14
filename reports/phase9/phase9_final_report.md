# Phase 9 — Model Evaluation and Improvement Report

Report generated: 2026-10-12 21:10:38

## 1. Objective
Phase 9 evaluates class imbalance handling, compares baseline and class-weighted XGBoost, investigates validation errors, and studies confidence-based alert policies.

## 2. Model Selection
The selected candidate was chosen using validation macro F1, with balanced accuracy as a supporting metric. The final test set was not used to select the Step 9.5 candidate.

### Step 9.5 — Validation Comparison

            Unnamed: 0  accuracy  balanced_accuracy  macro_precision  macro_recall  macro_f1
      Baseline XGBoost  0.997432           0.793809         0.894536      0.793809  0.826737
Class-Weighted XGBoost  0.996380           0.905721         0.900091      0.905721  0.886902

### Step 9.6 — Selection Record

```json
{
  "phase": 9,
  "step": "9.6",
  "selection_method": "Highest validation macro F1; balanced accuracy used as tie-breaker",
  "selected_candidate": "Class-Weighted XGBoost",
  "selected_metrics": {
    "macro_f1": 0.8869017644368561,
    "balanced_accuracy": 0.9057208235871612,
    "macro_recall": 0.9057208235871612
  },
  "selected_model_path": "D:\\smart-cybersecurity-ml\\models\\phase9_selected_xgboost_candidate.pkl",
  "final_test_set_used_for_selection": false,
  "status": "Selected for further validation"
}
```

## 3. Validation Error Analysis

### Classes with Lowest Recall

     class_name  support  precision   recall  f1_score
        rootkit        2   0.000000 0.000000  0.000000
           imap        2   1.000000 0.500000  0.666667
buffer_overflow        8   1.000000 0.625000  0.769231
       multihop        3   1.000000 0.666667  0.800000
          saint       15   0.857143 0.800000  0.827586
  snmpgetattack        7   0.333333 0.857143  0.480000
    warezmaster       48   0.870370 0.979167  0.921569
          mscan       50   1.000000 0.980000  0.989899
        ipsweep      652   0.990726 0.983129  0.986913
           back      189   1.000000 0.989418  0.994681

## 4. Class Imbalance Strategy Evaluation

                 model  accuracy  balanced_accuracy  macro_precision  macro_recall  macro_f1
      Baseline XGBoost  0.997432           0.793809         0.894536      0.793809  0.826737
Class-Weighted XGBoost  0.996380           0.905721         0.900091      0.905721  0.886902

### Rare-Class Comparison (Support < 50)

     class_name  support  baseline_recall  weighted_recall  baseline_f1  weighted_f1
        apache2       36         1.000000         1.000000     1.000000     1.000000
buffer_overflow        8         0.500000         0.625000     0.666667     0.769231
      ftp_write        1         1.000000         1.000000     1.000000     1.000000
     httptunnel        7         0.714286         1.000000     0.833333     1.000000
           imap        2         0.500000         0.500000     0.666667     0.666667
           land        3         0.000000         1.000000     0.000000     1.000000
     loadmodule        2         0.500000         1.000000     0.666667     0.500000
       mailbomb       15         1.000000         1.000000     1.000000     1.000000
       multihop        3         0.333333         0.666667     0.500000     0.800000
            pod       38         0.973684         1.000000     0.973684     0.962025
   processtable       34         1.000000         1.000000     1.000000     1.000000
        rootkit        2         0.000000         0.000000     0.000000     0.000000
          saint       15         0.666667         0.800000     0.740741     0.827586
  snmpgetattack        7         0.142857         0.857143     0.200000     0.480000
      snmpguess       16         1.000000         1.000000     1.000000     1.000000
    warezmaster       48         0.979167         0.979167     0.969072     0.921569

## 5. Confidence and Alert Policy Analysis

### Correct vs Incorrect Predictions

prediction_group  sample_count  mean_confidence
         Correct         23671         0.997326
       Incorrect            86         0.813687

### Threshold Comparison

 threshold  allow_count  alert_count  review_count  attack_detection_rate  normal_false_positive_count  normal_false_positive_rate
      0.50        12534        11215             8               0.998480                           49                    0.003897
      0.70        12509        11192            56               0.997586                           36                    0.002863
      0.90        12415        11152           190               0.995082                           24                    0.001909
      0.95        12300        11135           322               0.994009                           19                    0.001511
      0.99        11805        11045           907               0.986855                            9                    0.000716

### Alert Policy Summary

```json
{
  "step": "9.9",
  "analysis": "Threshold and alert policy simulation",
  "model": "Class-Weighted XGBoost",
  "validation_samples": 23757,
  "mean_confidence": 0.9966611266136169,
  "correct_predictions": 23671,
  "incorrect_predictions": 86,
  "default_metrics": {
    "accuracy": 0.9963800143115713,
    "balanced_accuracy": 0.9057208235871613,
    "macro_precision": 0.9000906333614147,
    "macro_recall": 0.9057208235871613,
    "macro_f1": 0.8869017644368561
  },
  "thresholds_evaluated": [
    0.5,
    0.7,
    0.9,
    0.95,
    0.99
  ],
  "final_test_set_used": false,
  "real_network_traffic_blocked": false,
  "status": "Threshold and alert policy analysis completed"
}
```

## 6. Main Findings

- Class-weighted XGBoost improved validation macro F1 and balanced accuracy compared with the baseline.
- Class weighting improved recall for several rare attacks, but did not solve every rare-class detection problem.
- The weighted model generated more false positives on normal traffic than the baseline in the reported validation comparison.
- Raising the alert confidence threshold reduced normal-traffic false positives, while also reducing the proportion of attacks flagged automatically and increasing review cases.
- Confidence scores are not necessarily calibrated probabilities and should not be treated as guaranteed measures of correctness.

## 7. Limitations

- Some rare classes have very few validation examples, so their precision and recall estimates are unstable.
- Duplicate feature patterns and eight previously identified overlapping rows with conflicting labels require careful interpretation; they do not by themselves prove data leakage.
- The final test set was evaluated during Phase 8. Therefore, although Steps 9.5–9.9 used validation data, the final test set should not be described as completely untouched by the overall project.
- The alert policy was simulated on validation predictions; no real network traffic was blocked.
- Results from the current dataset alone do not establish performance on CICIDS2017 or on real production traffic.

## 8. Conclusion

Class-weighted XGBoost is a promising candidate for improving minority-class performance on this validation split. Its higher false-positive count shows that a deployment policy must balance attack detection with alert workload. Further external validation and testing on additional datasets are needed before claiming production readiness.

## 9. Final Test Set Policy

Do not use the final test set for further threshold tuning or candidate selection. For stronger independent evidence, evaluate the finalized system on a new external dataset or a newly collected holdout set.