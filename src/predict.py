import joblib
import pandas as pd
from src.data_loader import ROOT

def load_model(): return joblib.load(ROOT/"models/lda_pipeline.joblib")
def predict_record(record, bundle=None):
    bundle=bundle or load_model(); features=bundle["features"]
    missing=set(features)-set(record); extra=set(record)-set(features)
    if missing: raise ValueError(f"Missing features: {', '.join(sorted(missing))}")
    if extra: raise ValueError(f"Unexpected features: {', '.join(sorted(extra))}")
    row=pd.DataFrame([{k:record[k] for k in features}],columns=features)
    out=int(bundle["pipeline"].predict(row)[0]); prob=bundle["pipeline"].predict_proba(row)[0]
    return {"prediction":"phishing" if out==1 else "legitimate","phishing_probability":float(prob[1]),"legitimate_probability":float(prob[0])}
