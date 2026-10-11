from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def load_config():
    return json.loads((ROOT / "config/experiment_config.json").read_text(encoding="utf-8"))

def load_dataset(path=None):
    path = Path(path) if path else ROOT / load_config()["dataset_path"]
    if not path.is_absolute(): path = ROOT / path
    if not path.exists(): raise FileNotFoundError(f"Dataset not found: {path}. Download phishing.csv and place it in data/.")
    df = pd.read_csv(path)
    cfg = load_config()
    target = cfg["target_column"]
    if target not in df: raise ValueError(f"Missing target column {target!r}")
    if set(df[target].dropna().unique()) != {-1, 1}: raise ValueError("Expected verified target labels {-1, 1}.")
    return df
