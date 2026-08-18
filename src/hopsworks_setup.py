"""
hopsworks_setup.py

One-time setup: connects to Hopsworks, creates a feature group, and
backfills it with your full historical engineered feature set.

Run from the project root:
    python src/hopsworks_setup.py
"""

import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Connect
# ============================================================

project = hopsworks.login(
    api_key_value=os.environ["HOPSWORKS_API_KEY"],
    project=os.environ["HOPSWORKS_PROJECT"],
    cert_folder="hopsworks_certs",   # ADD THIS LINE — relative folder in your project root
)

fs = project.get_feature_store()

# ============================================================
# Load your existing engineered features (historical backfill)
# ============================================================

df = pd.read_csv("data/processed/aqi_features.csv")
df["date"] = pd.to_datetime(df["date"])

# Hopsworks feature names must be lowercase letters/numbers/underscores only
# — no periods, no '+'. Rename before inserting.
df = df.rename(columns={
    "PM2.5": "pm2_5",
    "PM10": "pm10",
    "AQI_t+1": "aqi_t_plus_1",
    "AQI_t+2": "aqi_t_plus_2",
    "AQI_t+3": "aqi_t_plus_3",
    "PM2.5_rolling_mean_3": "pm2_5_rolling_mean_3",
    "PM2.5_rolling_mean_7": "pm2_5_rolling_mean_7",
})

print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# ============================================================
# Create (or get, if it already exists) the feature group
# ============================================================

aqi_fg = fs.get_or_create_feature_group(
    name="aqi_features",
    version=1,
    description="Karachi AQI historical + daily engineered features (28 model features + 3 targets)",
    primary_key=["date"],
    event_time="date",
    time_travel_format="HUDI",   # ADD THIS LINE — avoids needing the Delta library
)

# ============================================================
# Insert the backfill data
# ============================================================

aqi_fg.insert(df)

print("\nDone. Backfill inserted into Hopsworks feature group 'aqi_features' v1.")
print("Check your Hopsworks UI -> Feature Store -> aqi_features to confirm the rows landed.")