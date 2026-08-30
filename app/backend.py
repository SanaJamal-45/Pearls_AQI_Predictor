"""
app/backend.py

FastAPI backend for the Pearls AQI Predictor.

Run from the project root:
    uvicorn app.backend:app --reload --port 8000

Endpoints:
    GET  /health
    GET  /history?days=120
    GET  /predict/latest
    POST /predict            body: {"overrides": {"Temperature": 32, ...}}
    GET  /explain/latest?target=AQI_t+1
    POST /explain             body: {"target": "AQI_t+1", "overrides": {...}}
    GET  /explain/global?target=AQI_t+1&sample_size=200
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import TARGETS, aqi_category
from app.model_utils import (
    explain_global,
    explain_row,
    get_history,
    load_models,
    predict_all_horizons,
)

app = FastAPI(
    title="Pearls AQI Predictor API",
    description="Serves AQI forecasts (t+1/t+2/t+3) and SHAP-based feature "
                "importance explanations for the Karachi AQI predictor.",
    version="1.0.0",
)

# Wide-open CORS so the Streamlit dashboard (any host/port) can call this
# API directly from the browser. Tighten this to your dashboard's origin
# before deploying publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _json_safe(obj):
    """Recursively replace NaN/inf with None so the standard-library
    json encoder (which Starlette's JSONResponse uses, with allow_nan
    disabled) doesn't blow up on real-world gappy feature rows."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


class PredictRequest(BaseModel):
    overrides: Optional[dict] = None


class ExplainRequest(BaseModel):
    target: str = "AQI_t+1"
    overrides: Optional[dict] = None


@app.on_event("startup")
def _warm_up():
    # Trains/loads models once at startup instead of on the first request.
    load_models()


@app.get("/health")
def health():
    try:
        models = load_models()
        return {"status": "ok", "models_loaded": list(models.keys())}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/history")
def history(days: int = Query(120, ge=1, le=2000)):
    df = get_history(days)
    df = df.where(pd.notnull(df), None)
    return {
        "days": days,
        "data": [
            {"date": row["date"].strftime("%Y-%m-%d"), "AQI": row["AQI"]}
            for _, row in df.iterrows()
        ],
    }


def _format_predictions(predictions: dict) -> dict:
    out = {}
    for target, value in predictions.items():
        label, color = aqi_category(value)
        out[target] = {
            "predicted_aqi": round(value, 2),
            "category": label,
            "color": color,
        }
    return out


@app.get("/predict/latest")
def predict_latest():
    try:
        predictions, _ = predict_all_horizons()
        return {"predictions": _format_predictions(predictions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        predictions, X = predict_all_horizons(req.overrides)
        return _json_safe({
            "predictions": _format_predictions(predictions),
            "features_used": X.iloc[0].to_dict(),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/explain/latest")
def explain_latest(target: str = Query("AQI_t+1", enum=TARGETS)):
    try:
        _, X = predict_all_horizons()
        return _json_safe(explain_row(target, X))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
def explain(req: ExplainRequest):
    if req.target not in TARGETS:
        raise HTTPException(status_code=400, detail=f"target must be one of {TARGETS}")
    try:
        _, X = predict_all_horizons(req.overrides)
        return _json_safe(explain_row(req.target, X))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/explain/global")
def explain_global_endpoint(
    target: str = Query("AQI_t+1", enum=TARGETS),
    sample_size: int = Query(200, ge=10, le=2000),
):
    try:
        return _json_safe(explain_global(target, sample_size))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))