"""
hopsworks_model_registry.py

Pushes your 3 trained XGBoost models (t+1, t+2, t+3) into the Hopsworks
Model Registry, along with their validation metrics.

Run from the project root:
    python src/hopsworks_model_registry.py
"""

import os
import shutil
import joblib
import pandas as pd
from pathlib import Path
import hopsworks
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Connect (same fixes as the feature store script)
# ============================================================

project = hopsworks.login(
    api_key_value=os.environ["HOPSWORKS_API_KEY"],
    project=os.environ["HOPSWORKS_PROJECT"],
    cert_folder="hopsworks_certs",
)

mr = project.get_model_registry()

# ============================================================
# Load validation metrics to attach to each model
# ============================================================

results = pd.read_csv("reports/ml/xgboost_tuned_results.csv")

TARGETS = ["AQI_t+1", "AQI_t+2", "AQI_t+3"]
MODEL_FILES = {
    "AQI_t+1": "xgboost_tuned_AQI_tplus1.joblib",
    "AQI_t+2": "xgboost_tuned_AQI_tplus2.joblib",
    "AQI_t+3": "xgboost_tuned_AQI_tplus3.joblib",
}
REGISTRY_NAMES = {
    "AQI_t+1": "aqi_xgboost_t_plus_1",
    "AQI_t+2": "aqi_xgboost_t_plus_2",
    "AQI_t+3": "aqi_xgboost_t_plus_3",
}

# ============================================================
# Register each model
# ============================================================

for target in TARGETS:
    model_file = Path("models") / MODEL_FILES[target]

    # Hopsworks expects a directory containing the model artifact(s),
    # not a single loose file — stage each one in its own temp folder.
    staging_dir = Path("hopsworks_model_staging") / REGISTRY_NAMES[target]
    staging_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(model_file, staging_dir / model_file.name)

    test_row = results[(results["target"] == target) & (results["split"] == "test")].iloc[0]
    metrics = {
        "rmse": float(test_row["RMSE"]),
        "mae": float(test_row["MAE"]),
        "r2": float(test_row["R2"]),
    }

    py_model = mr.python.create_model(
        name=REGISTRY_NAMES[target],
        metrics=metrics,
        description=f"Tuned XGBoost model predicting {target} for Karachi AQI forecasting",
    )
    py_model.save(str(staging_dir))

    print(f"Registered {REGISTRY_NAMES[target]} with metrics: {metrics}")

print("\nAll 3 models registered. Check Hopsworks UI -> Model Registry to confirm.")