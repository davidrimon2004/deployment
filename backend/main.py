from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from model_utils import load_model_artifact, preprocess_input

app = FastAPI(title="Retail Sales Prediction API")

# Allow cross-origin calls from the UI (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SinglePrediction(BaseModel):
    store_id: str
    item_id: str | None = None
    sell_price: float | None = None
    wday: int | None = None
    is_event_day: int | None = None
    event_count: int | None = None
    lag_1: float | None = None
    lag_7: float | None = None
    wm_yr_wk: int | None = None
    snap: int | None = None
    year: int | None = None
    month: int | None = None
    day: int | None = None
    item_category: int | None = None
    item_subcategory: int | None = None
    item_number: int | None = None
    price_flag: int | None = None
    is_weekend: int | None = None
    snap_weekend: int | None = None
    wday_x_snap: int | None = None
    is_event: int | None = None
    event_impact: float | None = None


@app.post("/predict/single")
def predict_single(req: SinglePrediction):
    try:
        artifact = load_model_artifact(req.store_id)
    except Exception as e:
        raise HTTPException(404, str(e))

    try:
        df = pd.DataFrame([req.dict()])
        if "store_id" in df.columns:
            df = df.drop(columns=["store_id"])  # model does NOT take this as a feature

        X = preprocess_input(df, artifact)
        model = artifact["model"]

        pred = model.predict(X)[0]
        return {"prediction": int(np.round(pred))}
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    if "store_id" not in df.columns:
        raise HTTPException(400, "CSV must include store_id column")

    results = []

    for store, group in df.groupby("store_id"):
        artifact = load_model_artifact(store)
        X = preprocess_input(group, artifact)
        preds = np.round(artifact["model"].predict(X)).astype(int)

        for idx, p in zip(group.index, preds):
            results.append({"index": idx, "store": store, "prediction": int(p)})

    return {"predictions": results}


@app.get("/health")
def health():
    return {"status": "online"}
