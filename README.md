# Pearls AQI Predictor

Predicting Karachi's Air Quality Index (AQI) for the next 3 days using a serverless ML pipeline (automated data collection, feature engineering, model training, and an interactive dashboard with explainability and hazard alerts).

## Project Overview

This project builds an end-to-end ML system that:
- Collects live weather and pollutant data
- Engineers time-based and derived features (lags, rolling averages, change rates)
- Trains and evaluates multiple forecasting models
- Stores features and models in a feature store / model registry (Hopsworks)
- Runs on an automated hourly/daily schedule (GitHub Actions)
- Serves predictions through a FastAPI backend and an interactive Streamlit dashboard
- Explains predictions with SHAP and flags hazardous air quality

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML | scikit-learn, XGBoost, PyTorch |
| Feature Store / Model Registry | Hopsworks |
| CI/CD | GitHub Actions |
| Backend | FastAPI |
| Dashboard | Streamlit, Plotly |
| Explainability | SHAP |
| Data Sources | OpenWeather API (pollution + weather), historical Karachi AQI dataset |
| Version Control | Git / GitHub |

---

## 1. Feature Pipeline

- **`src/openweather_collector.py`** — fetches live pollutant (PM2.5, PM10, NO2, SO2, CO, O3) and weather (temperature, humidity, precipitation) forecast data from the OpenWeather API.
- **`src/feature_engineering.py`** — computes the full feature set from raw data:
  - Time-based: `day`, `day_of_week`, `month`, `year`, `day_of_year`
  - Derived: `AQI_lag_{1,2,3,7}`, `AQI_change_{1d,3d,7d}`, `AQI_rolling_mean_{3,7}`, `PM2.5_rolling_mean_{3,7}`, `PM10_rolling_mean_{3,7}`
- **`src/openweather_features.py`** — bridges OpenWeather's raw schema/scale into the exact 28-feature format the models expect. Since OpenWeather's own AQI index (1–5) is a different scale than the historical dataset's EPA-style AQI, this script derives AQI from PM2.5/PM10 concentrations using the standard EPA breakpoint formula instead of using OpenWeather's AQI field directly.

**Known limitation:** there is a 3-day gap (Aug 13–15, 2026) between where the historical dataset ends and where the live OpenWeather feed begins. Rows in this gap have partial/missing lag features; this does not block predictions for any date outside the gap.

## 2. Historical Data Backfill

- `data/raw/karachi_daily_aqi_weather.csv` — ~3.5 years of daily historical AQI, pollutant, and weather data for Karachi (1,319+ rows).
- Fully engineered into `data/processed/aqi_features.csv`, used for all model training and the Hopsworks feature store backfill.

## 3. Training Pipeline & Model Evaluation

Four model families were trained and evaluated on chronological train/validation/test splits (RMSE, MAE, R²):

| Model | t+1 R² | t+2 R² | t+3 R² |
|---|---|---|---|
| Ridge Regression | 0.844 | 0.32 | -0.09 |
| Random Forest | 0.79 | 0.36 | 0.26 |
| PyTorch NN | 0.38–0.57 | ~0.16 | ~-0.05 |
| **XGBoost (tuned)** | **0.851** | **0.363–0.41** | **0.106–0.18** |

**XGBoost was selected as the production model** — it matches or beats Ridge at t+1 and meaningfully improves the weaker t+2/t+3 horizons, which were the main accuracy problem in earlier iterations (Ridge's t+3 R² was *worse than guessing the average*).

Hyperparameters were tuned via `RandomizedSearchCV` (60 iterations, 3-fold CV) per forecast horizon. Feature importance confirms PM2.5 as the dominant driver (~36%), consistent with atmospheric science for PM2.5-driven urban air quality.

**PyTorch underperforms** — expected, given the dataset's validation set is only ~175 rows, too small for a neural net to outperform a well-regularized model.

## 4. Feature Store & Model Registry (Hopsworks)

- Feature group `aqi_features` (v1) — full historical backfill (1,319 rows, 32 columns) plus live daily inserts from the automated pipeline.
- Model Registry — `aqi_xgboost_t_plus_{1,2,3}`, versioned with RMSE/MAE/R² metrics attached to each version.

**Known limitations (documented honestly rather than hidden):**
- Hopsworks' Arrow Flight read service (`aqi_fg.read()`) failed consistently and reproducibly across multiple sessions with a server-side gRPC error. Writes (inserts, model registry uploads) worked reliably. **Workaround:** the daily retraining pipeline reads from local processed CSVs (kept in sync with the feature store) instead of reading back from Hopsworks, while still writing trained models to the Model Registry.
- The project's Hopsworks free-tier compute budget was exhausted mid-project. All feature store and model registry integration was fully built and verified working before this occurred (see commit history / screenshots). The dashboard and API were deliberately designed to serve from local files, so this did not block final delivery.

## 5. Automated CI/CD (GitHub Actions)

Two workflows in `.github/workflows/`:

- **`feature_pipeline.yml`** — runs hourly. Fetches live OpenWeather data → builds model-ready features → inserts into the Hopsworks feature store. *Verified working end-to-end.*
- **`training_pipeline.yml`** — runs daily. Retrains XGBoost on the latest available data → evaluates → pushes new model versions to the Hopsworks Model Registry. *Verified working end-to-end.*

Both can also be triggered manually via `workflow_dispatch` from the GitHub Actions tab.

## 6. Web Application

Two independent components, satisfying both the Streamlit/Gradio and Flask/FastAPI requirements:

- **`app/backend.py`** — FastAPI service exposing:
  - `GET /health`
  - `GET /history?days=N`
  - `GET /predict/latest`, `POST /predict` (supports "what-if" feature overrides)
  - `GET /explain/latest`, `POST /explain` — local SHAP explanation for a single prediction
  - `GET /explain/global` — global SHAP feature importance across a sample of historical rows

  Run with: `uvicorn app.backend:app --reload --port 8000`

- **`app/dashboard.py`** — Streamlit dashboard. Ships in a **standalone mode** that imports the prediction/SHAP logic directly from `app/model_utils.py` (no network call to the backend required), so it can run and deploy independently. Features:
  - Historical AQI trend chart with adjustable date range
  - 3-day forecast cards, color-coded by EPA AQI category
  - Hazardous air quality alert banner (triggers at "Unhealthy" and above)
  - "What-if" sliders (temperature, PM2.5, humidity) to explore hypothetical scenarios
  - SHAP explainability — both a per-prediction local explanation and an overall global feature-importance view

  Run with: `streamlit run app/dashboard.py`

## 7. Advanced Analytics

- **EDA** (`reports/eda/`) — distribution, seasonal, day-of-week, correlation, and lag-relationship analysis on the historical dataset.
- **SHAP explainability** — implemented via `shap.TreeExplainer`, both local (per-forecast) and global (overall feature importance), integrated directly into the dashboard.
- **Hazardous AQI alerts** — dashboard displays a warning/danger banner whenever any of the 3 forecast horizons cross into "Unhealthy" territory or worse.
- **Multiple forecasting models** — from linear (Ridge) through tree ensembles (Random Forest, XGBoost) to deep learning (PyTorch), with a documented, data-driven comparison rather than picking one model by default.
