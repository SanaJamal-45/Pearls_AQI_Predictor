"""
retrain_from_hopsworks.py

Pulls the full feature set from Hopsworks, retrains all 3 XGBoost
models, evaluates them, and pushes new versions to the Model Registry.
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
)

fs = project.get_feature_store()
mr = project.get_model_registry()
aqi_fg = fs.get_feature_group(name="aqi_features", version=1)

df = aqi_fg.read(read_options={"use_hive": True})

FEATURES = [
    "aqi", "pm2_5", "pm10", "no2", "so2", "co", "o3",
    "temperature", "humidity", "precipitation",
    "day", "day_of_week", "month", "year", "day_of_year",
    "aqi_lag_1", "aqi_lag_2", "aqi_lag_3", "aqi_lag_7",
    "aqi_change_1d", "aqi_change_3d", "aqi_change_7d",
    "aqi_rolling_mean_3", "aqi_rolling_mean_7",
    "pm2_5_rolling_mean_3", "pm2_5_rolling_mean_7",
    "pm10_rolling_mean_3", "pm10_rolling_mean_7",
]
TARGETS = {
    "aqi_t_plus_1": "aqi_xgboost_t_plus_1",
    "aqi_t_plus_2": "aqi_xgboost_t_plus_2",
    "aqi_t_plus_3": "aqi_xgboost_t_plus_3",
}
BEST_PARAMS = {
    "aqi_t_plus_1": dict(n_estimators=500, max_depth=2, learning_rate=0.05,
                          subsample=1.0, colsample_bytree=1.0,
                          reg_alpha=0.5, reg_lambda=5.0, min_child_weight=3),
    "aqi_t_plus_2": dict(n_estimators=300, max_depth=2, learning_rate=0.01,
                          subsample=0.6, colsample_bytree=0.7,
                          reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5),
    "aqi_t_plus_3": dict(n_estimators=300, max_depth=2, learning_rate=0.01,
                          subsample=0.6, colsample_bytree=0.7,
                          reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5),
}

staging_root = Path("hopsworks_model_staging")

for target_col, registry_name in TARGETS.items():
    # Only rows with a known outcome can be used for training
    clean = df.dropna(subset=[target_col] + FEATURES)

    X = clean[FEATURES]
    y = clean[target_col]
    clean = clean.sort_values("date")  # make sure it's chronological first
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
        description=f"Daily retrained XGBoost model for {target_col}",
    )
    py_model.save(str(staging_dir))

print("Retraining complete, all models pushed as new versions.")