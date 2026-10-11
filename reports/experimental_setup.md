# Experimental setup

Dataset: local Kaggle Phishing Website Detector CSV, audited in `dataset_profile.md`. Features are tabular encoded URL, page, domain and reputation properties, not raw URLs. Target `-1` phishing is mapped to positive 1; `+1` legitimate maps to 0. `Index` is excluded as a unique row identifier.

The implementation uses a stratified 80/20 split with seed 42. Standard scaling and median imputation are encapsulated in a scikit-learn `Pipeline` and refit inside stratified 5-fold training-only CV. LDA candidates are SVD and LSQR with automatic shrinkage; F1 selects the configuration from CV scores. The selected pipeline is fitted on all training data and evaluated once on test data. Timing separates fit time from an in-memory batch `predict` plus `predict_proba` on the held-out samples; throughput is samples divided by batch elapsed time. These local measurements do not establish production real-time performance.

```text
Dataset → Validation → Preprocessing → Stratified Split → LDA Training
        → Evaluation → Saved Model and Results → FastAPI Localhost Interface
```

Run `python -m src.train`; model goes to `models/lda_pipeline.joblib`, machine-readable metrics to `outputs/results.json` and `outputs/results.csv`, comparison to `outputs/lda_comparison.csv`, coefficients to `outputs/coefficients.csv`, and plots to `app/static/plots/`. Web requests load those artifacts and do not retrain.

Environment inspected: Windows, Python 3.14.2; installed versions include pandas 3.0.0, scikit-learn 1.9.1, FastAPI 0.142.2, Uvicorn 0.54.0, Matplotlib 3.11.1, and pytest 9.1.0. Training was attempted but scikit-learn failed importing a compiled `_loss` extension because Windows Application Control blocked it; therefore no model or measured evaluation results were generated. The localhost application itself started successfully; `/`, `/api/dataset`, and `/dataset` returned HTTP 200 and the dataset API reported 11,054 rows and 32 columns. Pytest collection found the same blocked extension in the training/prediction tests. Re-run in a compatible Python environment with working scikit-learn wheels. CPU model and RAM were not retrieved, so timing hardware specifics remain unavailable.

Limitations: random splitting cannot establish future-time or cross-domain generalization; raw URLs/domain identifiers are not present to group entities. Features may embed external reputation signals and do not represent a URL-only browser-time capture pipeline. The dataset is a fixed benchmark snapshot, and calibration and drift are outside this implementation.
