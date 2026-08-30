"""
app/model_utils.py

Loads the trained XGBoost models (the same models produced by
src/xgboost_tuned.py / pushed to Hopsworks by
src/hopsworks_model_registry.py) and wraps them with:

  - prediction helpers
  - SHAP explainers, for both a single-row "why did the model say
    this" explanation and a global "what matters most overall" view

If the .joblib files aren't present yet (they're gitignored), this
module trains them on the fly from the committed processed data so
the API/dashboard work out of the box after a fresh clone.
"""

from __future__ import annotations

import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from app.config import (
    FEATURES,
    HISTORY_PATH,
    LIVE_FEATURES_PATH,
    MODEL_DIR,
    MODEL_FILES,
    SPLITS_DIR,
    TARGETS,
)


# ============================================================
# Bootstrap: make sure models + splits exist
# ============================================================

def _run(script: str):
    print(f"[model_utils] running {script} ...")
    subprocess.run([sys.executable, script], check=True, cwd=str(MODEL_DIR.parent))


def ensure_models_exist():
    """Train the models the first time the app runs somewhere that
    doesn't already have models/*.joblib (they're gitignored)."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not SPLITS_DIR.exists() or not (SPLITS_DIR / "train.csv").exists():
        if not HISTORY_PATH.exists():
            _run("src/feature_engineering.py")
        _run("src/split_data.py")

    missing = [t for t in TARGETS if not MODEL_FILES[t].exists()]
    if missing:
        _run("src/xgboost_tuned.py")


# ============================================================
# Loading
# ============================================================

@lru_cache(maxsize=1)
def load_models() -> dict:
    ensure_models_exist()
    models = {}
    for target in TARGETS:
        path = MODEL_FILES[target]
        if not path.exists():
            raise FileNotFoundError(
                f"Model for {target} not found at {path}. "
                f"Run `python src/xgboost_tuned.py` from the project root."
            )
        models[target] = joblib.load(path)
    return models


@lru_cache(maxsize=1)
def load_explainers() -> dict:
    """One shap.TreeExplainer per horizon. TreeExplainer is exact and
    fast for XGBoost — no sampling / approximation needed."""
    models = load_models()
    return {target: shap.TreeExplainer(model) for target, model in models.items()}


@lru_cache(maxsize=1)
def load_background_sample() -> pd.DataFrame:
    """A fixed sample of real rows, used to compute global feature
    importance (mean |SHAP value| across many rows)."""
    ensure_models_exist()
    test_path = SPLITS_DIR / "test.csv"
    train_path = SPLITS_DIR / "train.csv"
    df = pd.read_csv(test_path if test_path.exists() else train_path)
    df = df.dropna(subset=FEATURES)
    return df[FEATURES].tail(200).reset_index(drop=True)


# ============================================================
# Latest feature row (for "predict tomorrow's AQI right now")
# ============================================================

def get_latest_feature_row() -> pd.Series:
    """Best available row of features for making a live prediction."""
    if LIVE_FEATURES_PATH.exists():
        df = pd.read_csv(LIVE_FEATURES_PATH)
    else:
        df = pd.read_csv(HISTORY_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df.iloc[-1]


def get_history(days: int = 120) -> pd.DataFrame:
    df = pd.read_csv(HISTORY_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df[["date", "AQI"]].tail(days).reset_index(drop=True)


# ============================================================
# Prediction
# ============================================================

def build_feature_vector(overrides: dict | None = None) -> pd.DataFrame:
    """Latest known row, with any user-supplied overrides applied
    (used for the dashboard's 'what-if' sliders)."""
    row = get_latest_feature_row()
    values = {f: row.get(f, np.nan) for f in FEATURES}
    if overrides:
        for k, v in overrides.items():
            if k in values:
                values[k] = v
    return pd.DataFrame([values], columns=FEATURES)


def predict_all_horizons(overrides: dict | None = None) -> dict:
    models = load_models()
    X = build_feature_vector(overrides)
    predictions = {}
    for target, model in models.items():
        pred = float(model.predict(X)[0])
        predictions[target] = pred
    return predictions, X


# ============================================================
# SHAP explanations
# ============================================================

def explain_row(target: str, X: pd.DataFrame) -> dict:
    """Local explanation: how much did each feature push this one
    prediction up or down from the model's average output?"""
    explainers = load_explainers()
    explainer = explainers[target]

    shap_values = explainer.shap_values(X)[0]
    base_value = float(explainer.expected_value)

    contributions = sorted(
        (
            {"feature": f, "value": float(X.iloc[0][f]) if pd.notna(X.iloc[0][f]) else None,
             "shap_value": float(sv)}
            for f, sv in zip(FEATURES, shap_values)
        ),
        key=lambda d: abs(d["shap_value"]),
        reverse=True,
    )

    return {
        "target": target,
        "base_value": base_value,
        "prediction": base_value + float(np.sum(shap_values)),
        "contributions": contributions,
    }


def explain_global(target: str, sample_size: int = 200) -> dict:
    """Global explanation: average |SHAP value| per feature across a
    sample of real historical rows — 'what matters most, overall'."""
    explainers = load_explainers()
    explainer = explainers[target]

    background = load_background_sample().tail(sample_size)
    shap_values = explainer.shap_values(background)

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = sorted(
        (
            {"feature": f, "mean_abs_shap": float(v)}
            for f, v in zip(FEATURES, mean_abs)
        ),
        key=lambda d: d["mean_abs_shap"],
        reverse=True,
    )

    return {
        "target": target,
        "n_samples": len(background),
        "importance": importance,
    }