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


# ============================================================
# Health Advisory
# ============================================================

ADVISORY_RULES = [
    {
        "max_aqi": 50,
        "level": "Good",
        "color": "#22c55e",
        "icon": "\u2705",
        "summary": "Air quality is satisfactory. Enjoy outdoor activities freely.",
        "groups": {
            "general": "No health risk. Enjoy outdoor activities.",
            "children": "Safe for all outdoor play and sports.",
            "elderly": "No precautions needed.",
            "respiratory": "No restrictions on outdoor activities.",
            "cardiac": "No restrictions. Air quality poses no risk.",
        },
        "actions": [
            "Open windows for natural ventilation",
            "Safe for all outdoor exercise",
            "No air purifier needed",
        ],
    },
    {
        "max_aqi": 100,
        "level": "Moderate",
        "color": "#eab308",
        "icon": "\u26a0\ufe0f",
        "summary": "Acceptable air quality. Unusually sensitive people should limit prolonged outdoor exertion.",
        "groups": {
            "general": "Most people face no risk. Enjoy the outdoors.",
            "children": "Safe for outdoor play. No restrictions.",
            "elderly": "No restrictions for most elderly people.",
            "respiratory": "Those with asthma may notice mild symptoms. Keep medication handy.",
            "cardiac": "No significant risk. Normal activities are fine.",
        },
        "actions": [
            "Unusually sensitive people should reduce prolonged outdoor exertion",
            "General public: no restrictions",
            "Ventilation is fine for most homes",
        ],
    },
    {
        "max_aqi": 150,
        "level": "Unhealthy for Sensitive Groups",
        "color": "#f97316",
        "icon": "\U0001f6a8",
        "summary": "Sensitive groups may experience health effects. General public is less likely to be affected.",
        "groups": {
            "general": "Most people unaffected. Monitor conditions if active outdoors.",
            "children": "Limit prolonged outdoor play. Schools should reduce outdoor recess.",
            "elderly": "Reduce outdoor exertion. Stay in filtered air when possible.",
            "respiratory": "Avoid prolonged outdoor activity. Use prescribed inhalers prophylactically.",
            "cardiac": "Avoid prolonged outdoor exertion. Watch for chest discomfort.",
        },
        "actions": [
            "Sensitive groups: reduce prolonged outdoor exertion",
            "Consider wearing KN95/N95 masks outdoors",
            "Close windows, use air purifier if available",
            "Children should switch to indoor activities",
        ],
    },
    {
        "max_aqi": 200,
        "level": "Unhealthy",
        "color": "#ef4444",
        "icon": "\U0001f6a8",
        "summary": "Everyone may begin to experience health effects. Sensitive groups face serious risks.",
        "groups": {
            "general": "Reduce prolonged outdoor exertion. Take breaks indoors.",
            "children": "Avoid all outdoor sports and play. Keep children indoors.",
            "elderly": "Stay indoors. Use air conditioning or purifiers.",
            "respiratory": "EMERGENCY: Stay indoors. Ensure rescue medications are accessible.",
            "cardiac": "Avoid all outdoor exertion. Seek medical attention for symptoms.",
        },
        "actions": [
            "Everyone: avoid prolonged outdoor exertion",
            "Wear KN95/N95 masks if going outside",
            "Close all windows. Run air purifiers on high.",
            "Cancel outdoor events and sports",
            "Monitor symptoms: coughing, throat irritation, shortness of breath",
        ],
    },
    {
        "max_aqi": 300,
        "level": "Very Unhealthy",
        "color": "#9333ea",
        "icon": "\U0001f6d1",
        "summary": "Health alert: significant risk of health effects for everyone.",
        "groups": {
            "general": "\U0001f6d1 Avoid ALL outdoor activities. Stay in filtered air.",
            "children": "\U0001f6d1 EMERGENCY: Children must stay indoors. No outdoor activities.",
            "elderly": "\U0001f6d1 Critical risk. Stay indoors with air purification.",
            "respiratory": "\U0001f6d1 EMERGENCY: Stay indoors. Have emergency action plan ready.",
            "cardiac": "\U0001f6d1 EMERGENCY: Avoid all physical exertion. Call doctor if symptoms appear.",
        },
        "actions": [
            "\U0001f6d1 EMERGENCY: Avoid ALL outdoor physical activity",
            "\U0001f6d1 Wear N95 masks if outdoor exposure is unavoidable",
            "\U0001f6d1 Run all available air purifiers continuously",
            "\U0001f6d1 Keep windows and doors sealed",
            "Seek medical help for breathing difficulties",
        ],
    },
    {
        "max_aqi": 500,
        "level": "Hazardous",
        "color": "#7e0023",
        "icon": "\u2620\ufe0f",
        "summary": "HAZARDOUS: Emergency conditions. Entire population is affected.",
        "groups": {
            "general": "EMERGENCY: Do NOT go outside under any circumstance.",
            "children": "EMERGENCY: Children are at extreme risk. Stay indoors.",
            "elderly": "EMERGENCY: Extreme health risk. Evacuate if possible.",
            "respiratory": "EMERGENCY: Risk of severe medical events. Seek hospital care.",
            "cardiac": "EMERGENCY: Risk of heart attack/stroke. Seek immediate care.",
        },
        "actions": [
            "EMERGENCY: Stay indoors at all times",
            "Wear N95/HEPA-filtered masks if any outdoor exposure",
            "Seal rooms and run air purifiers at maximum",
            "Consider temporary evacuation to areas with cleaner air",
            "Call emergency services for breathing difficulties",
        ],
    },
]


def _get_advisory(aqi: float) -> dict:
    for rule in ADVISORY_RULES:
        if aqi <= rule["max_aqi"]:
            return rule
    return ADVISORY_RULES[-1]


@app.get("/health-advisory")
def health_advisory():
    try:
        predictions, _ = predict_all_horizons()
        advisories = {}
        for target, value in predictions.items():
            advisory = _get_advisory(float(value))
            advisories[target] = {
                "predicted_aqi": round(float(value), 2),
                "level": advisory["level"],
                "color": advisory["color"],
                "icon": advisory["icon"],
                "summary": advisory["summary"],
                "groups": advisory["groups"],
                "actions": advisory["actions"],
            }
        return _json_safe({"advisories": advisories})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))