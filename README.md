# LDA phishing website classification

Research implementation of scikit-learn Linear Discriminant Analysis (LDA) with a FastAPI localhost demonstration. The verified source file in this workspace is `data/phishing.csv`.

## Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.train
pytest
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. Install a supported Python/scikit-learn wheel if Windows blocks compiled package extensions. Training writes the model to `models/lda_pipeline.joblib`, results and comparison tables to `outputs/`, and plots to `app/static/plots/`. Evaluation numbers are never hardcoded into the app.

If dataset file is missing, download the Kaggle Phishing Website Detector CSV and place it at `data/phishing.csv`. The included dataset is structured feature data, not raw URLs. Label `-1` is phishing and `+1` is legitimate. `Index` is a unique sequential identifier and is excluded from predictors. Train/test splitting is stratified 80/20, seed 42; preprocessing is fitted within each training fold/pipeline. Configurations are chosen using 5-fold training-only F1, then evaluated once on the held-out test partition.

## Project map

- `src/`: loading, preprocessing, training and prediction.
- `app/`: FastAPI application and static/templates.
- `config/experiment_config.json`: experiment settings.
- `reports/`: dataset audit, methodology, research review, setup and demo guide.
- `paper/`: paper subsection and BibTeX.
- `outputs/`, `models/`: generated local experiment artifacts (ignored by git).

`python -m src.train` is explicit; HTTP page loads do not retrain the model. This project demonstrates static tabular classification and does not fetch or inspect live websites. Dataset age, distribution shift, feature extraction validity and correlated indicators limit claims about deployment performance.
