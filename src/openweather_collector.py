import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

LAT = 24.8607
LON = 67.0011
CITY = "Karachi"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "openweather"

POLLUTION_OUTPUT = OUTPUT_DIR / "pollution_forecast_raw.csv"
WEATHER_OUTPUT = OUTPUT_DIR / "weather_forecast_raw.csv"

TIMEZONE = "Asia/Karachi"


# ============================================================
# LOAD API KEY
# ============================================================

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not API_KEY:
    raise ValueError(
        "OPENWEATHER_API_KEY not found.\n"
        "Make sure your .env file contains:\n\n"
        "OPENWEATHER_API_KEY=0485819569065d4dab438d9fac73a3b8"
    )


# ============================================================
# API URLs
# ============================================================

POLLUTION_URL = (
    "https://api.openweathermap.org/data/2.5/air_pollution/forecast"
)

WEATHER_URL = (
    "https://api.openweathermap.org/data/2.5/forecast"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FETCH POLLUTION FORECAST
# ============================================================

def fetch_pollution_forecast():

    print("\n" + "=" * 60)
    print("FETCHING AIR POLLUTION FORECAST")
    print("=" * 60)

    params = {
        "lat": LAT,
        "lon": LON,
        "appid": API_KEY
    }

    response = requests.get(
        POLLUTION_URL,
        params=params,
        timeout=30
    )

    print("Status code:", response.status_code)

    response.raise_for_status()

    data = response.json()

    records = data.get("list", [])

    if not records:
        raise ValueError(
            "No pollution forecast records returned."
        )

    rows = []

    collection_time = datetime.now().isoformat()

    for record in records:

        components = record.get("components", {})

        timestamp = pd.to_datetime(
            record["dt"],
            unit="s",
            utc=True
        )

        timestamp = timestamp.tz_convert(TIMEZONE)

        rows.append(
            {
                "city": CITY,
                "latitude": LAT,
                "longitude": LON,

                "datetime": timestamp,
                "date": timestamp.date(),

                "openweather_aqi": record["main"]["aqi"],

                "co": components.get("co"),
                "no": components.get("no"),
                "no2": components.get("no2"),
                "o3": components.get("o3"),
                "so2": components.get("so2"),
                "pm2_5": components.get("pm2_5"),
                "pm10": components.get("pm10"),
                "nh3": components.get("nh3"),

                "collected_at": collection_time
            }
        )

    df = pd.DataFrame(rows)

    df = df.sort_values("datetime")

    df.to_csv(
        POLLUTION_OUTPUT,
        index=False
    )

    print(f"Records collected: {len(df)}")
    print(
        f"Forecast range: "
        f"{df['datetime'].min()} → {df['datetime'].max()}"
    )

    print("\nPollution columns:")
    print(df.columns.tolist())

    print("\nFirst 5 records:")
    print(
        df.head().to_string(index=False)
    )

    print("\nSaved to:")
    print(POLLUTION_OUTPUT)

    return df


# ============================================================
# FETCH WEATHER FORECAST
# ============================================================

def fetch_weather_forecast():

    print("\n" + "=" * 60)
    print("FETCHING WEATHER FORECAST")
    print("=" * 60)

    params = {
        "lat": LAT,
        "lon": LON,
        "appid": API_KEY,
        "units": "metric"
    }

    response = requests.get(
        WEATHER_URL,
        params=params,
        timeout=30
    )

    print("Status code:", response.status_code)

    response.raise_for_status()

    data = response.json()

    records = data.get("list", [])

    if not records:
        raise ValueError(
            "No weather forecast records returned."
        )

    rows = []

    collection_time = datetime.now().isoformat()

    for record in records:

        timestamp = pd.to_datetime(
            record["dt"],
            unit="s",
            utc=True
        )

        timestamp = timestamp.tz_convert(TIMEZONE)

        main = record.get("main", {})
        wind = record.get("wind", {})
        clouds = record.get("clouds", {})
        rain = record.get("rain", {})

        rows.append(
            {
                "city": CITY,
                "latitude": LAT,
                "longitude": LON,

                "datetime": timestamp,
                "date": timestamp.date(),

                "temperature": main.get("temp"),
                "feels_like": main.get("feels_like"),
                "temperature_min": main.get("temp_min"),
                "temperature_max": main.get("temp_max"),

                "pressure": main.get("pressure"),
                "humidity": main.get("humidity"),
                "dew_point": main.get("dew_point"),

                "wind_speed": wind.get("speed"),
                "wind_direction": wind.get("deg"),
                "wind_gust": wind.get("gust"),

                "cloudiness": clouds.get("all"),

                "precipitation_probability": record.get(
                    "pop",
                    0
                ),

                "rain_3h": rain.get(
                    "3h",
                    0
                ),

                "visibility": record.get(
                    "visibility"
                ),

                "weather_id": (
                    record["weather"][0]["id"]
                    if record.get("weather")
                    else None
                ),

                "weather_main": (
                    record["weather"][0]["main"]
                    if record.get("weather")
                    else None
                ),

                "weather_description": (
                    record["weather"][0]["description"]
                    if record.get("weather")
                    else None
                ),

                "collected_at": collection_time
            }
        )

    df = pd.DataFrame(rows)

    df = df.sort_values("datetime")

    df.to_csv(
        WEATHER_OUTPUT,
        index=False
    )

    print(f"Records collected: {len(df)}")

    print(
        f"Forecast range: "
        f"{df['datetime'].min()} → {df['datetime'].max()}"
    )

    print("\nWeather columns:")
    print(df.columns.tolist())

    print("\nFirst 5 records:")
    print(
        df.head().to_string(index=False)
    )

    print("\nSaved to:")
    print(WEATHER_OUTPUT)

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("PEARLS AQI PREDICTOR")
    print("OPENWEATHER DATA COLLECTOR")
    print("=" * 60)

    print(f"\nCity: {CITY}")
    print(f"Latitude: {LAT}")
    print(f"Longitude: {LON}")
    print(f"Timezone: {TIMEZONE}")

    pollution_df = fetch_pollution_forecast()

    weather_df = fetch_weather_forecast()

    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)

    print(
        f"\nPollution records: {len(pollution_df)}"
    )

    print(
        f"Weather records: {len(weather_df)}"
    )

    print("\nFiles created:")

    print(POLLUTION_OUTPUT)
    print(WEATHER_OUTPUT)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()