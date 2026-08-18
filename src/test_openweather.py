import os
import requests
from dotenv import load_dotenv


# ==================================================
# Load API key
# ==================================================

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not API_KEY:
    raise ValueError(
        "OPENWEATHER_API_KEY not found. "
        "Make sure your .env file exists in the project root."
    )


# ==================================================
# Karachi coordinates
# ==================================================

LAT = 24.8607
LON = 67.0011


# ==================================================
# Test 1 — Current Air Pollution
# ==================================================

print("\n" + "=" * 60)
print("TEST 1 — CURRENT AIR POLLUTION")
print("=" * 60)

pollution_url = "https://api.openweathermap.org/data/2.5/air_pollution"

params = {
    "lat": LAT,
    "lon": LON,
    "appid": API_KEY
}

response = requests.get(
    pollution_url,
    params=params,
    timeout=30
)

print("Status code:", response.status_code)

if response.status_code != 200:
    print("Response:")
    print(response.text)
else:
    data = response.json()

    print("\nPollution response:")
    print(data)


# ==================================================
# Test 2 — Air Pollution Forecast
# ==================================================

print("\n" + "=" * 60)
print("TEST 2 — AIR POLLUTION FORECAST")
print("=" * 60)

forecast_url = "https://api.openweathermap.org/data/2.5/air_pollution/forecast"

response = requests.get(
    forecast_url,
    params=params,
    timeout=30
)

print("Status code:", response.status_code)

if response.status_code != 200:
    print("Response:")
    print(response.text)
else:
    data = response.json()

    print("\nNumber of forecast records:")

    if "list" in data:
        print(len(data["list"]))

        print("\nFirst forecast record:")
        print(data["list"][0])


# ==================================================
# Test 3 — Weather Forecast
# ==================================================

print("\n" + "=" * 60)
print("TEST 3 — WEATHER FORECAST")
print("=" * 60)

weather_url = "https://api.openweathermap.org/data/2.5/forecast"

weather_params = {
    "lat": LAT,
    "lon": LON,
    "appid": API_KEY,
    "units": "metric"
}

response = requests.get(
    weather_url,
    params=weather_params,
    timeout=30
)

print("Status code:", response.status_code)

if response.status_code != 200:
    print("Response:")
    print(response.text)
else:
    data = response.json()

    print("\nCity:")
    print(data.get("city", {}).get("name"))

    print("\nNumber of forecast records:")

    if "list" in data:
        print(len(data["list"]))

        print("\nFirst forecast record:")
        print(data["list"][0])


# ==================================================
# Complete
# ==================================================

print("\n" + "=" * 60)
print("OPENWEATHER API TEST COMPLETE")
print("=" * 60)