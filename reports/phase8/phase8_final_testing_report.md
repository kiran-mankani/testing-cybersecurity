# Phase 8 — Model Testing and Validation Report

Generated: 2026-10-12 20:00:44

## 1. Test Dataset Overview

- Total test samples: 29,696
- Number of classes evaluated: 28
- Correct predictions: 29,623
- Incorrect predictions: 73
- Test dataset distribution: Original imbalanced distribution retained.
- Test samples were not rebalanced during this evaluation.

## 2. Overall Model Performance

| Metric | Actual Result |
|---|---:|
| Accuracy | 0.9975 |
| Balanced Accuracy | 0.8408 |
| Macro Precision | 0.8442 |
| Macro Recall | 0.8408 |
| Macro F1-score | 0.8370 |
| Weighted F1-score | 0.9974 |
| Matthews Correlation Coefficient | N/A |
| Cohen's Kappa | 0.9960 |

Note: These values are read from the saved Step 8.5 metrics JSON.
If a metric is shown as N/A, verify its key name in that JSON file.

## 3. Rare-Class Testing

- Rare-class threshold: fewer than 50 test samples.
- Number of rare classes: 15.
- Zero-recall classes in the full test report: ftp_write, multihop, rootkit.
- Zero-recall classes among rare classes: ftp_write, multihop, rootkit.

Rare classes require careful interpretation because very small test supports
can make recall and F1 estimates unstable.

## 4. Five Lowest-F1 Classes

| Class | Test_Samples | Precision | Recall | F1_Score |
| --- | --- | --- | --- | --- |
| ftp_write | 2 | 0.0 | 0.0 | 0.0 |
| multihop | 3 | 0.0 | 0.0 | 0.0 |
| rootkit | 3 | 0.0 | 0.0 | 0.0 |
| saint | 19 | 0.6111 | 0.5789 | 0.5946 |
| loadmodule | 2 | 1.0 | 0.5 | 0.6667 |

## 5. Confidence Testing

- Mean confidence: 0.9983659982681274
- Median confidence: 0.9999880790710448
- Minimum confidence: 0.4087581038475036
- Maximum confidence: 0.99999737739563
- Mean confidence on correct predictions: 0.9988763332366944
- Mean confidence on incorrect predictions: 0.7912572026252747
- Predictions above or equal to 90% confidence: 29583.0
- Predictions below 60% confidence: 26.0

High confidence does not guarantee a correct prediction. Probability calibration
and high-confidence errors should be evaluated separately.

## 6. Simulated Threat Response

| Response_Action | Sample_Count |
| --- | --- |
| ALLOW | 15676 |
| ALERT | 13907 |
| REVIEW | 87 |
| INVESTIGATE | 26 |

These response actions were simulated for evaluation only.
No live network traffic was blocked, modified, or otherwise controlled.

The correctness rate within a response group measures prediction correctness,
not the security or safety of the response policy.

## 7. Limitations and Recommendations

1. Overall accuracy is high, but macro F1 and balanced accuracy indicate
   weaker performance across some classes.
2. Investigate zero-recall classes, including ftp_write, multihop, and rootkit.
3. Investigate other weaker classes such as saint and snmpgetattack.
4. Review high-confidence incorrect predictions and assess probability calibration.
5. Examine false negatives before considering automated threat responses.
6. Verify train/test separation and preprocessing consistency before deployment.
7. Avoid claims of reliable detection for every attack class based only on
   a small number of test examples.

## 8. Conclusion

The model achieved strong overall performance on the evaluated test dataset.
However, rare-class detection, probability calibration, and the safety of
automated response policies require additional validation.

These results alone do not establish production readiness.
