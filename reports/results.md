# Experimental results

Generated from `python -m src.train` using a stratified grouped five-fold split. Identical predictor vectors are kept in one group across both the held-out partition and training-only cross-validation. The first group fold is the approximately 80/20 holdout; this dataset yielded 8,844 training rows and 2,210 test rows.

The selected configuration is scikit-learn LDA with the SVD solver. Selection used grouped five-fold training-only F1; mean F1 was 0.90677 (fold standard deviation 0.00757). LSQR with automatic shrinkage had the same recorded mean and standard deviation, so the candidate ranking is a tie and the configured candidate order selects SVD.

| Test metric | Value |
|---|---:|
| Accuracy | 0.9290 |
| Precision | 0.9237 |
| Recall | 0.9152 |
| F1 | 0.9194 |
| Specificity | 0.9399 |
| ROC AUC | 0.9758 |
| Average precision | 0.9712 |

Confusion counts, with phishing positive: TN 1,157; FP 74; FN 83; TP 896. These estimates are for this grouped split of the fixed dataset and do not establish performance on future websites, domains, or a live feature pipeline. Runtime figures in `outputs/results.json` are local single-run timings.
