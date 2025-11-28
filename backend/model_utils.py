import os
import pickle
import pandas as pd
import numpy as np
import requests
from pathlib import Path

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
_model_cache = {}

# Define the expected features for the model
EXPECTED_FEATURES = [
    'wm_yr_wk', 'wday', 'snap', 'year', 'month', 'day', 'lag_1',
    'item_category', 'item_subcategory', 'item_number', 'sell_price',
    'price_flag', 'is_weekend', 'snap_weekend', 'wday_x_snap', 'lag_7',
    'is_event', 'event_count', 'event_impact'
]

def load_model_artifact(store_id: str):
    """Load a model artifact for a specific store."""
    global _model_cache
    if store_id in _model_cache:
        return _model_cache[store_id]

    file_path = MODELS_DIR / f"{store_id}.pkl"
    if not file_path.exists():
        download_err = None
        # 1) Try MODEL_STORE_URL if configured
        base = os.environ.get("MODEL_STORE_URL")
        if base:
            try:
                url = f"{base.rstrip('/')}/{store_id}.pkl"
                resp = requests.get(url, stream=True, timeout=30)
                if resp.status_code == 200:
                    MODELS_DIR.mkdir(parents=True, exist_ok=True)
                    with open(file_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                else:
                    raise FileNotFoundError(f"Model not found at remote URL: {url} (status {resp.status_code})")
            except Exception as e:
                download_err = e

        # 2) Try Kaggle if MODEL_STORE_URL didn't yield a file
        if not file_path.exists():
            kaggle_dataset = os.environ.get("KAGGLE_DATASET")
            if kaggle_dataset:
                # try kagglehub first (preferred if available)
                try:
                    import kagglehub
                    path = kagglehub.model_download(kaggle_dataset)
                    candidate = Path(path) / f"{store_id}.pkl"
                    if candidate.exists():
                        MODELS_DIR.mkdir(parents=True, exist_ok=True)
                        candidate.replace(file_path)
                    else:
                        # scan for the model file in the downloaded folder
                        for p in Path(path).rglob(f"{store_id}.pkl"):
                            MODELS_DIR.mkdir(parents=True, exist_ok=True)
                            p.replace(file_path)
                            break
                except Exception:
                    # try kaggle CLI fallback using models instances versions download
                    try:
                        import subprocess
                        MODELS_DIR.mkdir(parents=True, exist_ok=True)
                        # Use kaggle models instances versions download for the specific version
                        cmd = [
                            "kaggle", "models", "instances", "versions", "download",
                            kaggle_dataset, "-p", str(MODELS_DIR)
                        ]
                        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
                    except Exception as e:
                        # record failure
                        download_err = e

        # if after attempts the file still doesn't exist, raise
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found and download attempts failed: {download_err}")

    with open(file_path, "rb") as f:
        loaded = pickle.load(f)

    # Handle case where the file contains just the model (not a dict artifact)
    if isinstance(loaded, dict):
        artifact = loaded
    else:
        # Wrap the model in an artifact dict
        artifact = {"model": loaded, "features": None}

    if "model" not in artifact:
        raise ValueError("Model artifact must contain a 'model' key.")

    _model_cache[store_id] = artifact
    return artifact


def preprocess_input(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Preprocess incoming data according to stored training preprocessing."""
    df = df.copy()
    
    # Use the expected features for this model
    features = EXPECTED_FEATURES
    
    # Ensure every feature exists
    for col in features:
        if col not in df:
            df[col] = 0

    # Convert all columns to numeric, handling non-numeric types
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to numeric, using 0 as default for non-numeric values
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        # Ensure numeric types are int or float
        elif df[col].dtype.kind not in ['i', 'f', 'b']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    X = df[features]

    # Ensure X has exactly the right columns in the right order
    X = X[EXPECTED_FEATURES]

    # Apply scaler if available
    if artifact.get("scaler") is not None:
        scaler = artifact["scaler"]
        num_cols = [c for c in features if X[c].dtype.kind in "fi"]
        if num_cols:
            X[num_cols] = scaler.transform(X[num_cols])

    # Apply item encoder if available
    if artifact.get("item_encoder") is not None and "item_id" in df:
        enc = artifact["item_encoder"]
        df_enc = enc.transform(df[["item_id"]])
        df["item_id_enc"] = df_enc
        if "item_id_enc" in features:
            X["item_id_enc"] = df["item_id_enc"]

    return X
