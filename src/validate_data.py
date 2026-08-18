import pandas as pd
from pathlib import Path


# ==================================================
# 1. File path
# ==================================================

DATA_PATH = Path("data/raw/karachi_daily_aqi_weather.csv")


# ==================================================
# 2. Load data
# ==================================================

df = pd.read_csv(DATA_PATH)


# ==================================================
# 3. Basic dataset information
# ==================================================

print("\n" + "=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nColumn names:")
print(df.columns.tolist())


# ==================================================
# 4. Data types
# ==================================================

print("\n" + "=" * 50)
print("DATA TYPES")
print("=" * 50)

print(df.dtypes)


# ==================================================
# 5. Missing values
# ==================================================

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)

print(df.isnull().sum())


# ==================================================
# 6. Duplicate rows
# ==================================================

print("\n" + "=" * 50)
print("DUPLICATE ROWS")
print("=" * 50)

print(f"Duplicate rows: {df.duplicated().sum()}")


# ==================================================
# 7. Date validation
# ==================================================

df["date"] = pd.to_datetime(df["date"])

print("\n" + "=" * 50)
print("DATE INFORMATION")
print("=" * 50)

print(f"Start date: {df['date'].min()}")
print(f"End date:   {df['date'].max()}")

print(f"Duplicate dates: {df['date'].duplicated().sum()}")

print(
    f"Dates sorted: "
    f"{df['date'].is_monotonic_increasing}"
)


# ==================================================
# 8. Date gaps
# ==================================================

sorted_dates = df["date"].sort_values()

date_difference = sorted_dates.diff().dt.days

gaps = date_difference[date_difference > 1]

print("\n" + "=" * 50)
print("DATE GAPS")
print("=" * 50)

print(f"Number of gaps: {len(gaps)}")

if len(gaps) > 0:
    print("\nGaps found:")
    print(gaps)


# ==================================================
# 9. Numerical columns
# ==================================================

numeric_columns = [
    "AQI",
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "O3",
    "Temperature",
    "Humidity",
    "Precipitation"
]


# ==================================================
# 10. Negative values
# ==================================================

print("\n" + "=" * 50)
print("NEGATIVE VALUES")
print("=" * 50)

for column in numeric_columns:

    negative_count = (df[column] < 0).sum()

    print(f"{column}: {negative_count}")


# ==================================================
# 11. Basic statistics
# ==================================================

print("\n" + "=" * 50)
print("BASIC STATISTICS")
print("=" * 50)

print(df[numeric_columns].describe())


# ==================================================
# 12. Validation complete
# ==================================================

print("\n" + "=" * 50)
print("VALIDATION COMPLETE")
print("=" * 50)