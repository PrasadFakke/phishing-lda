# Experimental setup

Dataset: local Kaggle Phishing Website Detector CSV, audited in `dataset_profile.md`. Features are tabular encoded URL, page, domain and reputation properties, not raw URLs. Target `-1` phishing is mapped to positive 1; `+1` legitimate maps to 0. `Index` is excluded as a unique row identifier.

Identical predictor vectors are grouped, including vectors observed with conflicting labels. `StratifiedGroupKFold` with five folds and seed 42 supplies the approximately 80/20 held-out partition (the first fold) and the five training-only CV folds. This prevents the same feature vector from crossing a partition boundary. The local run produced 8,844 training rows and 2,210 held-out rows. Standard scaling is fitted within each pipeline fold; no imputation is used because the audited dataset has no missing values. LDA candidates are SVD and LSQR with automatic shrinkage; grouped CV F1 selects the candidate, then the selected pipeline is evaluated once on the holdout.

Timing separates fit time from an in-memory batch `predict` plus `predict_proba` on held-out samples; throughput is samples divided by batch elapsed time. These local measurements do not establish production real-time performance. Run `python -m src.train`; the fitted model, JSON/CSV metrics, comparison, coefficients and plots are written under `models/`, `outputs/` and `app/static/plots/`.

The recorded environment is Windows 11, Python 3.14.2, scikit-learn 1.9.1. The latest local run completed and produced the results in `outputs/results.json`. Earlier notes about a blocked compiled extension describe a prior attempt and are no longer the current run status.

Limitations: feature-vector grouping reduces duplicate leakage but cannot establish future-time or cross-domain generalization; raw URLs/domain identifiers are not present. Features may embed external reputation signals and do not represent a URL-only browser-time capture pipeline. The dataset is a fixed benchmark snapshot, and calibration and drift remain outside this implementation.
