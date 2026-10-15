\# Final Report — Cybersecurity Model Review



\## Main Model



\*\*Standard Weighted XGBoost\*\* has been selected as the main model.



\*\*Reason:\*\* Missing an attack is more dangerous than a false alarm. Standard Weighted XGBoost reduces missed attacks (86 vs 140), even though false alarms increase slightly (309 vs 112).



\---



\## Fair Comparison Results (Cleaned Data)



Three XGBoost models were fairly compared. After 5-fold cross-validation on deduplicated data, the following results were obtained:



| Metric | Original XGBoost | Standard Weighted XGBoost | Targeted Weighted XGBoost |

|--------|------------------|---------------------------|---------------------------|

| \*\*Accuracy\*\* | 99.73% | \*\*99.62%\*\* | 99.61% |

| \*\*Macro F1\*\* | 0.867 | \*\*0.859\*\* | 0.858 |

| \*\*Missed Attacks\*\* | 140 | \*\*86\*\* | 82 |

| \*\*False Alarms\*\* | 112 | \*\*309\*\* | 324 |

| \*\*Wrong Predictions\*\* | 355 | \*\*504\*\* | 515 |



\*\*Main Model:\*\* Standard Weighted XGBoost

\- \*\*Accuracy:\*\* 99.62%

\- \*\*Macro F1:\*\* 0.859

\- \*\*Missed Attacks:\*\* 86

\- \*\*False Alarms:\*\* 309



\---



\## Important Note



These numbers are taken from `results/fair\_comparison/cv\_summary.csv`, which is based on \*\*cleaned (deduplicated) data\*\*.



\*\*The earlier overlapping test set numbers (99.73% accuracy) are not reported\*\* because they were not fair — 17.1% of test rows were copies of training rows.



\---



\## New Attack Results



\## New Attack Results



When tested on the official NSL-KDD split (train on training file, test on test file), all three models catch about 93% of known attack types but only 24-29% of new attack types:



| Model | Known Attack Types Caught | New Attack Types Caught |

|-------|---------------------------|-------------------------|

| Original XGBoost | 93.4% | 23.6% |

| \*\*Standard Weighted XGBoost\*\* | \*\*93.4%\*\* | \*\*28.0%\*\* |

| Targeted Weighted XGBoost | 93.4% | 29.4% |



\*\*Test Set Info:\*\*

\- Total Attacks: 11,299

\- New Attack Types: 1,146 (never seen in training)

\- Normal Traffic: 11,245



This is the honest measure of how a model would cope with a brand-new attack. About 1 in 4 new attack types is caught.



\*\*To improve this:\*\*

1\. Add anomaly detection

2\. Test on CICIDS2017 dataset

\---



\## Recommendations



1\. ✅ \*\*Keep Standard Weighted XGBoost\*\* as the main model

2\. ✅ \*\*Drop Targeted Weighted XGBoost\*\*

3\. ✅ \*\*Report cleaned numbers\*\* (99.62% accuracy, 0.859 macro F1)

4\. ⏳ Quote the new-attack result (24-29%)

5\. ⏳ Add anomaly detection

6\. ⏳ Test on CICIDS2017

7\. ⏳ Group the rarest attacks

8\. ⏳ Re-run the notebooks



\---



\*\*Date:\*\* 15 Oct 2026

\*\*Source:\*\* `results/fair\_comparison/cv\_summary.csv`

