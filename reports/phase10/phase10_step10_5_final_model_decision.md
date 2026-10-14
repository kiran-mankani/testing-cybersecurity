# Phase 10 — Step 10.5: Final Model Decision

## 1. Audit status

Step 10.4 passed all recorded consistency checks.
This confirms consistency of saved results, not independent model validity.

## 2. Overall model comparison

| Metric | Original XGBoost | Class-Weighted XGBoost |
|---|---:|---:|
| Accuracy | 99.75% | 99.64% |
| Balanced accuracy | 84.08% | 86.52% |
| Macro precision | 84.42% | 82.47% |
| Macro recall | 84.08% | 86.52% |
| Macro F1 | 83.70% | 83.46% |
| Weighted F1 | 99.74% | 99.65% |
| Incorrect predictions | 73 | 107 |

## 3. Interpretation

- Original XGBoost accuracy: 99.75%.
- Class-Weighted XGBoost accuracy: 99.64%.
- Original XGBoost balanced accuracy: 84.08%.
- Class-Weighted XGBoost balanced accuracy: 86.52%.
- Original XGBoost macro F1: 83.70%.
- Class-Weighted XGBoost macro F1: 83.46%.

The original model has higher overall accuracy and macro F1.
The weighted model has higher balanced accuracy and macro recall.
The weighted model therefore is not automatically superior on every metric.

## 4. Classes requiring further attention

The following classes have recall below 80% for the weighted model:

- ftp_write: recall 50.00%, support 2.
- loadmodule: recall 50.00%, support 2.
- multihop: recall 0.00%, support 3.
- rootkit: recall 0.00%, support 3.
- saint: recall 68.42%, support 19.
- snmpgetattack: recall 77.78%, support 9.

## 5. Model decision

No universal winner. Original XGBoost performs better on overall accuracy and macro F1, while Class-Weighted XGBoost performs better on balanced accuracy and macro recall. Select a deployment candidate only after deciding the acceptable trade-off between missed attacks and false-positive alerts.

## 6. Limitations

The test set was evaluated in Phase 8 and reused in Phase 10. These results are not an independent evaluation on a new untouched test set.

Offline classification results do not prove real-time network detection or automatic threat prevention.

Before real deployment, evaluate on a new independent dataset and validate performance on realistic network traffic.
