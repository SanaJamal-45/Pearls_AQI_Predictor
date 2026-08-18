import pandas as pd
from pathlib import Path


# ==================================================
# Paths
# ==================================================

INPUT_PATH = Path("data/raw/karachi_daily_aqi_weather.csv")
OUTPUT_PATH = Path("data/processed/aqi_features.csv")


# ==================================================
# Load data
# ==================================================

df = pd.read_csv(INPUT_PATH)

df["date"] = pd.to_datetime(df["date"])


# ==================================================
# Remove redundant target
# ==================================================

df = df.drop(columns=["Next_Day_AQI"])


# ==================================================
# Sort chronologically
# ==================================================

df = df.sort_values("date").reset_index(drop=True)


# ==================================================
# AQI lookup by date
# ==================================================

aqi_lookup = df.set_index("date")["AQI"]


# ==================================================
# Future AQI targets
# ==================================================

df["AQI_t+1"] = aqi_lookup.reindex(
    df["date"] + pd.Timedelta(days=1)
).to_numpy()

df["AQI_t+2"] = aqi_lookup.reindex(
    df["date"] + pd.Timedelta(days=2)
).to_numpy()

df["AQI_t+3"] = aqi_lookup.reindex(
    df["date"] + pd.Timedelta(days=3)
).to_numpy()


# ==================================================
# Time-based features
# ==================================================

df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df["day_of_year"] = df["date"].dt.dayofyear


# ==================================================
# Historical AQI lag features
# ==================================================

df["AQI_lag_1"] = aqi_lookup.reindex(
    df["date"] - pd.Timedelta(days=1)
).to_numpy()

df["AQI_lag_2"] = aqi_lookup.reindex(
    df["date"] - pd.Timedelta(days=2)
).to_numpy()

df["AQI_lag_3"] = aqi_lookup.reindex(
    df["date"] - pd.Timedelta(days=3)
).to_numpy()

df["AQI_lag_7"] = aqi_lookup.reindex(
    df["date"] - pd.Timedelta(days=7)
).to_numpy()


# ==================================================
# AQI change features
# ==================================================

df["AQI_change_1d"] = (
    df["AQI"] - df["AQI_lag_1"]
)

df["AQI_change_3d"] = (
    df["AQI"] - df["AQI_lag_3"]
)

df["AQI_change_7d"] = (
    df["AQI"] - df["AQI_lag_7"]
)


# ==================================================
# Rolling historical features
#
# IMPORTANT:
# These use actual calendar dates.
# Missing dates therefore do NOT get treated
# as if they were consecutive observations.
# ==================================================

def date_based_rolling_mean(
    lookup,
    dates,
    days
):
    """
    Calculate a rolling mean using the previous
    `days` calendar days, excluding the current day.
    """

    values = []

    for current_date in dates:

        start_date = (
            current_date -
            pd.Timedelta(days=days)
        )

        end_date = (
            current_date -
            pd.Timedelta(days=1)
        )

        historical_values = lookup.loc[
            (lookup.index >= start_date)
            &
            (lookup.index <= end_date)
        ]

        if len(historical_values) == 0:
            values.append(float("nan"))
        else:
            values.append(
                historical_values.mean()
            )

    return values


# ==================================================
# AQI rolling means
# ==================================================

df["AQI_rolling_mean_3"] = (
    date_based_rolling_mean(
        aqi_lookup,
        df["date"],
        3
    )
)

df["AQI_rolling_mean_7"] = (
    date_based_rolling_mean(
        aqi_lookup,
        df["date"],
        7
    )
)


# ==================================================
# Pollutant lookup tables
# ==================================================

pm25_lookup = df.set_index("date")["PM2.5"]

pm10_lookup = df.set_index("date")["PM10"]


# ==================================================
# PM2.5 rolling means
# ==================================================

df["PM2.5_rolling_mean_3"] = (
    date_based_rolling_mean(
        pm25_lookup,
        df["date"],
        3
    )
)

df["PM2.5_rolling_mean_7"] = (
    date_based_rolling_mean(
        pm25_lookup,
        df["date"],
        7
    )
)


# ==================================================
# PM10 rolling means
# ==================================================

df["PM10_rolling_mean_3"] = (
    date_based_rolling_mean(
        pm10_lookup,
        df["date"],
        3
    )
)

df["PM10_rolling_mean_7"] = (
    date_based_rolling_mean(
        pm10_lookup,
        df["date"],
        7
    )
)


# ==================================================
# Save processed dataset
# ==================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ==================================================
# Summary
# ==================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 60)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ==================================================
# Display new features
# ==================================================

print("\nNew features:")

new_features = [
    "AQI_change_3d",
    "AQI_change_7d",
    "AQI_rolling_mean_3",
    "AQI_rolling_mean_7",
    "PM2.5_rolling_mean_3",
    "PM2.5_rolling_mean_7",
    "PM10_rolling_mean_3",
    "PM10_rolling_mean_7"
]

print(
    df[new_features]
    .head(10)
    .to_string(index=False)
)


# ==================================================
# Check missing-date region
# ==================================================

print("\nTargets and rolling features around "
      "the missing-date region:")

gap_start = pd.Timestamp("2026-06-05")
gap_end = pd.Timestamp("2026-06-10")

columns_to_show = [
    "date",
    "AQI",
    "AQI_t+1",
    "AQI_t+2",
    "AQI_t+3",
    "AQI_lag_1",
    "AQI_lag_3",
    "AQI_rolling_mean_3",
    "AQI_rolling_mean_7"
]

print(
    df.loc[
        df["date"].between(
            gap_start,
            gap_end
        ),
        columns_to_show
    ].to_string(index=False)
)


# ==================================================
# Last rows
# ==================================================

print("\nLast 5 rows:")

print(
    df[
        [
            "date",
            "AQI",
            "AQI_t+1",
            "AQI_t+2",
            "AQI_t+3"
        ]
    ]
    .tail()
    .to_string(index=False)
)


# ==================================================
# Save confirmation
# ==================================================

print("\nSaved to:")
print(OUTPUT_PATH)

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 60)