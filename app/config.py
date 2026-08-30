"""
app/config.py

Shared configuration for the FastAPI backend and Streamlit dashboard.
Keeping this in one place means the API and the UI can never disagree
about which features/targets/models are in play.
"""

from pathlib import Path

# ============================================================
# Paths (all relative to the project root — run everything
# from the repo root, e.g. `uvicorn app.backend:app`)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

HISTORY_PATH = DATA_DIR / "aqi_features.csv"          # has targets, full history
LIVE_FEATURES_PATH = DATA_DIR / "model_ready_features.csv"  # latest rows, no targets
SPLITS_DIR = DATA_DIR / "splits"                       # train/validation/test.csv

# ============================================================
# Features / targets — must match src/xgboost_tuned.py exactly,
# since that's the script that produced the .joblib files below.
# ============================================================

FEATURES = [
    "AQI", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
    "Temperature", "Humidity", "Precipitation",
    "day", "day_of_week", "month", "year", "day_of_year",
    "AQI_lag_1", "AQI_lag_2", "AQI_lag_3", "AQI_lag_7",
    "AQI_change_1d", "AQI_change_3d", "AQI_change_7d",
    "AQI_rolling_mean_3", "AQI_rolling_mean_7",
    "PM2.5_rolling_mean_3", "PM2.5_rolling_mean_7",
    "PM10_rolling_mean_3", "PM10_rolling_mean_7",
]

TARGETS = ["AQI_t+1", "AQI_t+2", "AQI_t+3"]

MODEL_FILES = {
    "AQI_t+1": MODEL_DIR / "xgboost_tuned_AQI_tplus1.joblib",
    "AQI_t+2": MODEL_DIR / "xgboost_tuned_AQI_tplus2.joblib",
    "AQI_t+3": MODEL_DIR / "xgboost_tuned_AQI_tplus3.joblib",
}

# ============================================================
# AQI category bands (US EPA breakpoints) — used to color the
# dashboard and give predictions a human-readable label.
# ============================================================

AQI_CATEGORIES = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#ffff00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 500, "Hazardous", "#7e0023"),
]


def aqi_category(value: float):
    """Return (label, color) for a given AQI value."""
    if value is None:
        return "Unknown", "#999999"
    for low, high, label, color in AQI_CATEGORIES:
        if low <= value <= high:
            return label, color
    if value > 500:
        return "Hazardous", "#7e0023"
    return "Good", "#00e400"