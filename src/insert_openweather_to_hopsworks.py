"""
insert_openweather_to_hopsworks.py

Takes the freshly-generated model_ready_features.csv (from
openweather_features.py) and inserts it into the same Hopsworks
feature group as the historical backfill.
"""

import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv

load_dotenv()

project = hopsworks.login(
    api_key_value=os.environ["HOPSWORKS_API_KEY"],
    project=os.environ["HOPSWORKS_PROJECT"],
)

fs = project.get_feature_store()
aqi_fg = fs.get_feature_group(name="aqi_features", version=1)

df = pd.read_csv("data/processed/model_ready_features.csv")
df["date"] = pd.to_datetime(df["date"])

# Same renames as the historical backfill — must match the feature
# group's existing schema exactly
df = df.rename(columns={
    "PM2.5": "pm2_5",
    "PM10": "pm10",
    "PM2.5_rolling_mean_3": "pm2_5_rolling_mean_3",
    "PM2.5_rolling_mean_7": "pm2_5_rolling_mean_7",
})

# Live rows have no known future AQI yet — add the target columns as
# null so the schema matches. They'll stay null in the feature store;
# that's fine, they're only used for training on already-past rows.
for col in ["aqi_t_plus_1", "aqi_t_plus_2", "aqi_t_plus_3"]:
    if col not in df.columns:
        df[col] = None

aqi_fg.insert(df)
print(f"Inserted {len(df)} live rows into aqi_features.")