"""
retrain_local_fallback.py

Same retraining logic as retrain_from_hopsworks.py, but reads from
your local files instead of Hopsworks' read service (which is
currently having a server-side issue). Still pushes the trained
models to the Hopsworks Model Registry — only the READ step is local,
the WRITE (model registry) still goes to Hopsworks and works fine.
"""

import os
import shutil
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import hopsworks
from dotenv import load_dotenv

load_dotenv()

project = hopsworks.login(
    api_key_value=os.environ["HOPSWORKS_API_KEY"],
    project=os.environ["HOPSWORKS_PROJECT"],
    cert_folder="hopsworks_certs",
)
mr = project.get_model_registry()

# ---- Read locally instead of aqi_fg.read() ----
historical = pd.read_csv("data/processed/aqi_features.csv")
live = pd.read_csv("data/processed/model_ready_features.csv")

# Historical has the target columns (AQI_t+1/2/3), live doesn't (future
# unknown) — only historical rows are usable for training anyway.
df = historical.copy()

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
TARGETS = {
    "AQI_t+1": "aqi_xgboost_t_plus_1",
    "AQI_t+2": "aqi_xgboost_t_plus_2",
    "AQI_t+3": "aqi_xgboost_t_plus_3",
}
BEST_PARAMS = {
    "AQI_t+1": dict(n_estimators=500, max_depth=2, learning_rate=0.05,
                     subsample=1.0, colsample_bytree=1.0,
                     reg_alpha=0.5, reg_lambda=5.0, min_child_weight=3),
    "AQI_t+2": dict(n_estimators=300, max_depth=2, learning_rate=0.01,
                     subsample=0.6, colsample_bytree=0.7,
                     reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5),
    "AQI_t+3": dict(n_estimators=300, max_depth=2, learning_rate=0.01,
                     subsample=0.6, colsample_bytree=0.7,
                     reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5),
}

staging_root = Path("hopsworks_model_staging")

for target_col, registry_name in TARGETS.items():
    clean = df.dropna(subset=[target_col] + FEATURES)
    X = clean[FEATURES]
    y = clean[target_col]
    clean = clean.sort_values("date")
    split_point = int(len(clean) * 0.85)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    model = XGBRegressor(random_state=42, n_jobs=-1, **BEST_PARAMS[target_col])
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }
    print(f"{registry_name}: {metrics}")

    staging_dir = staging_root / registry_name
    staging_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, staging_dir / f"{registry_name}.joblib")

    py_model = mr.python.create_model(
        name=registry_name,
        metrics=metrics,
        description=f"Retrained XGBoost model for {target_col} (local data fallback)",
    )
    py_model.save(str(staging_dir))

print("\nRetraining complete (local data), models pushed to Hopsworks registry as new versions.")