# Phase 7 — Deep Model Evaluation Report

**Generated:** 2026-10-12 15:25:17

## 1. Project Overview

This report evaluates the trained XGBoost intrusion-detection model using overall classification metrics, per-class performance, ranking metrics, confidence analysis, and misclassification analysis.

## 2. Overall Performance

| Metric | Value |
|---|---:|
| Accuracy | 0.997542 |
| Balanced Accuracy | 0.840815 |
| Macro Precision | 0.844194 |
| Macro Recall | 0.840815 |
| Macro F1 | 0.836976 |
| Weighted F1 | 0.997449 |
| Matthews Correlation Coefficient | 0.995989 |
| Log Loss | 0.009923 |
| Top-3 Accuracy | 0.999731 |

- **Test samples:** 29,696
- **Number of classes:** 28

## 3. Strongest Classes

| Class | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| httptunnel | 1.0000 | 1.0000 | 1.0000 | 8 |
| snmpguess | 1.0000 | 1.0000 | 1.0000 | 20 |
| back | 1.0000 | 1.0000 | 1.0000 | 237 |
| mailbomb | 1.0000 | 1.0000 | 1.0000 | 19 |
| imap | 1.0000 | 1.0000 | 1.0000 | 2 |
| apache2 | 1.0000 | 1.0000 | 1.0000 | 46 |
| neptune | 0.9998 | 0.9999 | 0.9998 | 9574 |
| smurf | 1.0000 | 0.9984 | 0.9992 | 637 |
| normal | 0.9983 | 0.9989 | 0.9986 | 15718 |
| portsweep | 1.0000 | 0.9970 | 0.9985 | 660 |

## 4. Weakest Classes

| Class | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| ftp_write | 0.0000 | 0.0000 | 0.0000 | 2 |
| multihop | 0.0000 | 0.0000 | 0.0000 | 3 |
| rootkit | 0.0000 | 0.0000 | 0.0000 | 3 |
| saint | 0.6111 | 0.5789 | 0.5946 | 19 |
| loadmodule | 1.0000 | 0.5000 | 0.6667 | 2 |
| snmpgetattack | 0.6364 | 0.7778 | 0.7000 | 9 |
| land | 0.5714 | 1.0000 | 0.7273 | 4 |
| buffer_overflow | 1.0000 | 0.8889 | 0.9412 | 9 |
| warezmaster | 0.9500 | 0.9500 | 0.9500 | 60 |
| mscan | 0.9828 | 0.9194 | 0.9500 | 62 |

## 5. Rare Classes with Weak F1

The following classes have fewer than 100 test samples and an F1 score below 0.50. These thresholds are used for investigation, not as universal definitions of poor performance.

| Class | F1 Score | Support | Recall |
|---|---:|---:|---:|
| ftp_write | 0.0000 | 2 | 0.0000 |
| multihop | 0.0000 | 3 | 0.0000 |
| rootkit | 0.0000 | 3 | 0.0000 |

## 6. Prediction and Confidence Analysis

- Correct predictions: **29,623**
- Incorrect predictions: **73**
- Total predictions analyzed: **29,696**
- Accuracy recalculated from prediction details: **0.997542**
- High-confidence incorrect predictions: **31**

High-confidence errors deserve review because the model assigned high confidence to an incorrect prediction. Confidence alone does not establish whether an intrusion was successfully prevented.

## 7. Frequent Misclassification Patterns

| Actual Class | Predicted Class | Error Count |
|---|---|---:|
| saint | satan | 7 |
| satan | saint | 7 |
| normal | satan | 4 |
| normal | snmpgetattack | 4 |
| nmap | ipsweep | 4 |
| satan | normal | 3 |
| normal | pod | 3 |
| normal | land | 3 |
| warezmaster | normal | 3 |
| teardrop | normal | 2 |

## 8. ROC-AUC and Average Precision

Refer to `phase7_roc_auc_summary.json`, `phase7_precision_recall_summary.json`, and the dashboard class-metrics CSV for the detailed ranking metrics. These metrics can help assess how well classes are separated or ranked, but do not replace precision, recall, and F1.

## 9. Evaluation Dashboard

The visual summary is saved as `results/phase7/phase7_evaluation_dashboard.png`.

## 10. Limitations and Next Steps

- Evaluate weak and rare classes individually; overall accuracy can hide poor minority-class performance.
- Investigate high-confidence errors and the most frequent actual-to-predicted class confusions.
- Validate on appropriately separated, representative data before making deployment claims.
- A classification result is not proof that a threat was blocked. Prevention requires an integrated response mechanism and separate testing of its actions.

## 11. Conclusion

The evaluated model achieved an accuracy of 0.9975, macro F1 of 0.8370, and weighted F1 of 0.9974 on the evaluated test set. Per-class results and error analysis should be considered alongside these overall metrics when assessing intrusion-detection performance.
