"""
openweather_features.py

Bridges raw OpenWeather forecast data (hourly, OpenWeather's own schema
and AQI scale) into the exact 28-feature format your model was trained on
(historical Karachi AQI schema, EPA-style AQI scale).

Run from the project root:
    python src/openweather_features.py
"""

import pandas as pd
from pathlib import Path

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HISTORICAL_PATH = PROJECT_ROOT / "data" / "raw" / "karachi_daily_aqi_weather.csv"
POLLUTION_PATH = PROJECT_ROOT / "data" / "raw" / "openweather" / "pollution_forecast_raw.csv"
WEATHER_PATH = PROJECT_ROOT / "data" / "raw" / "openweather" / "weather_forecast_raw.csv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "model_ready_features.csv"

HISTORY_SEED_DAYS = 14  # trailing history kept only to seed lag_7 / rolling_mean_7


# ============================================================
# EPA AQI calculation from raw concentrations
#
# OpenWeather's own `aqi` field is a 1-5 qualitative index and is NOT
# compatible with your historical AQI column's scale. Instead we
# calculate a standard EPA-style AQI from PM2.5 and PM10 (µg/m³),
# taking the max of the two sub-indices (how official AQI works when
# multiple pollutants are available).
#
# NOTE: approximation. No overlapping day exists between historical
# data and the OpenWeather feed, so this can't be directly calibrated
# against the original source.
# ============================================================

PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500),
]
PM10_BREAKPOINTS = [
    (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
    (255, 354, 151, 200), (355, 424, 201, 300),
    (425, 504, 301, 400), (505, 604, 401, 500),
]


def _epa_subindex(concentration, breakpoints):
    if pd.isna(concentration):
        return float("nan")
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= concentration <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (concentration - c_lo) + i_lo
    if concentration > breakpoints[-1][1]:
        return breakpoints[-1][3]
    return float("nan")


def calculate_aqi(pm25, pm10):
    aqi_pm25 = _epa_subindex(pm25, PM25_BREAKPOINTS)
    aqi_pm10 = _epa_subindex(pm10, PM10_BREAKPOINTS)
    return max(v for v in [aqi_pm25, aqi_pm10] if not pd.isna(v))


# ============================================================
# Load and aggregate OpenWeather data to daily
# ============================================================

def load_openweather_daily():
    pollution = pd.read_csv(POLLUTION_PATH)
    weather = pd.read_csv(WEATHER_PATH)

    pollution["date"] = pd.to_datetime(pollution["date"])
    weather["date"] = pd.to_datetime(weather["date"])

    pollution_daily = pollution.groupby("date").agg(
        {"pm2_5": "mean", "pm10": "mean", "no2": "mean",
         "so2": "mean", "co": "mean", "o3": "mean"}
    ).reset_index()

    weather_daily = weather.groupby("date").agg(
        {"temperature": "mean", "humidity": "mean", "rain_3h": "sum"}
    ).reset_index()

    # Inner join: only keep dates where BOTH pollution and weather exist.
    # OpenWeather's two endpoints don't always cover identical date
    # ranges — an outer join creates rows with missing pollutant data
    # and no valid AQI.
    daily = pollution_daily.merge(weather_daily, on="date", how="inner")

    daily = daily.rename(columns={
        "pm2_5": "PM2.5", "pm10": "PM10", "no2": "NO2", "so2": "SO2",
        "co": "CO", "o3": "O3", "temperature": "Temperature",
        "humidity": "Humidity", "rain_3h": "Precipitation",
    })

    daily["AQI"] = daily.apply(
        lambda row: calculate_aqi(row["PM2.5"], row["PM10"]), axis=1
    )

    cols = ["date", "AQI", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
            "Temperature", "Humidity", "Precipitation"]
    return daily[cols].sort_values("date").reset_index(drop=True)


# ============================================================
# Date-based lag / rolling helper (same logic as feature_engineering.py)
# ============================================================

def date_based_rolling_mean(lookup, dates, days):
    values = []
    for current_date in dates:
        start_date = current_date - pd.Timedelta(days=days)
        end_date = current_date - pd.Timedelta(days=1)
        window = lookup.loc[(lookup.index >= start_date) & (lookup.index <= end_date)]
        values.append(window.mean() if len(window) else float("nan"))
    return values


# ============================================================
# Build combined feature set
# ============================================================

def build_model_ready_features():
    historical = pd.read_csv(HISTORICAL_PATH)
    historical["date"] = pd.to_datetime(historical["date"])
    historical = historical.drop(columns=["Next_Day_AQI"], errors="ignore")
    historical = historical.sort_values("date").reset_index(drop=True)

    seed = historical.tail(HISTORY_SEED_DAYS).copy()
    new_rows = load_openweather_daily()

    last_historical_date = historical["date"].max()
    first_new_date = new_rows["date"].min()
    gap_days = (first_new_date - last_historical_date).days - 1
    if gap_days > 0:
        print(
            f"WARNING: {gap_days} day(s) with no ground-truth AQI between "
            f"{last_historical_date.date()} and {first_new_date.date()}. "
            f"Lag/rolling features for the first new row(s) will be NaN or "
            f"based on a shorter window until this gap is backfilled."
        )

    combined = pd.concat([seed, new_rows], ignore_index=True)
    combined = combined.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)

    aqi_lookup = combined.set_index("date")["AQI"]
    pm25_lookup = combined.set_index("date")["PM2.5"]
    pm10_lookup = combined.set_index("date")["PM10"]

    combined["day"] = combined["date"].dt.day
    combined["day_of_week"] = combined["date"].dt.dayofweek
    combined["month"] = combined["date"].dt.month
    combined["year"] = combined["date"].dt.year
    combined["day_of_year"] = combined["date"].dt.dayofyear

    combined["AQI_lag_1"] = aqi_lookup.reindex(combined["date"] - pd.Timedelta(days=1)).to_numpy()
    combined["AQI_lag_2"] = aqi_lookup.reindex(combined["date"] - pd.Timedelta(days=2)).to_numpy()
    combined["AQI_lag_3"] = aqi_lookup.reindex(combined["date"] - pd.Timedelta(days=3)).to_numpy()
    combined["AQI_lag_7"] = aqi_lookup.reindex(combined["date"] - pd.Timedelta(days=7)).to_numpy()

    combined["AQI_change_1d"] = combined["AQI"] - combined["AQI_lag_1"]
    combined["AQI_change_3d"] = combined["AQI"] - combined["AQI_lag_3"]
    combined["AQI_change_7d"] = combined["AQI"] - combined["AQI_lag_7"]

    combined["AQI_rolling_mean_3"] = date_based_rolling_mean(aqi_lookup, combined["date"], 3)
    combined["AQI_rolling_mean_7"] = date_based_rolling_mean(aqi_lookup, combined["date"], 7)
    combined["PM2.5_rolling_mean_3"] = date_based_rolling_mean(pm25_lookup, combined["date"], 3)
    combined["PM2.5_rolling_mean_7"] = date_based_rolling_mean(pm25_lookup, combined["date"], 7)
    combined["PM10_rolling_mean_3"] = date_based_rolling_mean(pm10_lookup, combined["date"], 3)
    combined["PM10_rolling_mean_7"] = date_based_rolling_mean(pm10_lookup, combined["date"], 7)

    result = combined[combined["date"].isin(new_rows["date"])].reset_index(drop=True)

    feature_order = [
        "date", "AQI", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
        "Temperature", "Humidity", "Precipitation",
        "day", "day_of_week", "month", "year", "day_of_year",
        "AQI_lag_1", "AQI_lag_2", "AQI_lag_3", "AQI_lag_7",
        "AQI_change_1d", "AQI_change_3d", "AQI_change_7d",
        "AQI_rolling_mean_3", "AQI_rolling_mean_7",
        "PM2.5_rolling_mean_3", "PM2.5_rolling_mean_7",
        "PM10_rolling_mean_3", "PM10_rolling_mean_7",
    ]
    return result[feature_order]


if __name__ == "__main__":
    features = build_model_ready_features()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)

    print(f"\nRows: {len(features)}  Columns: {len(features.columns)}")
    print(features[["date", "AQI", "PM2.5", "PM10", "AQI_lag_1", "AQI_lag_7", "AQI_rolling_mean_7"]].to_string(index=False))

    null_counts = features.isna().sum()
    problem_cols = null_counts[null_counts > 0]
    if len(problem_cols):
        print("\nColumns with missing values:")
        print(problem_cols.to_string())
    print(f"\nSaved to: {OUTPUT_PATH}")