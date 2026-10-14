# Phase 10 — Step 10.11: Candidate Selection and Validation Decision

## Selected Candidate

`phase10_targeted_weight_xgboost_candidate.pkl`

## Decision Basis

- Higher validation balanced accuracy than standard class weighting
- Higher validation macro recall
- Higher validation macro F1
- Rootkit validation recall improved, but precision remains low

## Limitations

- Rootkit validation support is only 2 samples
- Rare-class false positives require further inspection
- The test set was already evaluated in Phase 8 and is not an untouched independent test set
- Offline classification results do not establish real-time threat prevention

## Decision Status

Selected for further evaluation; not production-approved.
