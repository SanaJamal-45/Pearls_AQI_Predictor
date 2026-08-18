# ============================================================
# PEARLS AQI PREDICTOR
# Phase 2 — Time-Based Train / Validation / Test Split
# ============================================================

import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path("data/processed/aqi_features.csv")

OUTPUT_DIR = Path("data/processed/splits")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(INPUT_PATH)

    df["date"] = pd.to_datetime(df["date"])

    # Make absolutely sure data is chronological
    df = df.sort_values("date").reset_index(drop=True)

    return df


# ============================================================
# DEFINE FEATURES AND TARGETS
# ============================================================

FEATURES = [
    "AQI",
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "O3",
    "Temperature",
    "Humidity",
    "Precipitation",
    "day",
    "day_of_week",
    "month",
    "year",
    "day_of_year",
    "AQI_lag_1",
    "AQI_lag_2",
    "AQI_lag_3",
    "AQI_lag_7",
    "AQI_change_1d"
]

TARGETS = [
    "AQI_t+1",
    "AQI_t+2",
    "AQI_t+3"
]


# ============================================================
# SPLIT FUNCTION
# ============================================================

def create_split(df):

    print("\n" + "=" * 60)
    print("CREATING TIME-BASED DATA SPLIT")
    print("=" * 60)

    # --------------------------------------------------------
    # Remove rows where required features are unavailable
    # --------------------------------------------------------

    df_model = df.dropna(
        subset=FEATURES
    ).copy()

    print(
        f"\nOriginal rows: {len(df)}"
    )

    print(
        f"Rows after feature cleaning: "
        f"{len(df_model)}"
    )

    # --------------------------------------------------------
    # Time-based split
    #
    # TRAIN      : before 2026-01-01
    # VALIDATION : 2026-01-01 → 2026-06-30
    # TEST       : 2026-07-01 onward
    # --------------------------------------------------------

    train = df_model[
        df_model["date"] < "2026-01-01"
    ].copy()

    validation = df_model[
        (df_model["date"] >= "2026-01-01") &
        (df_model["date"] < "2026-07-01")
    ].copy()

    test = df_model[
        df_model["date"] >= "2026-07-01"
    ].copy()

    return train, validation, test


# ============================================================
# SAVE SPLITS
# ============================================================

def save_splits(train, validation, test):

    train_path = OUTPUT_DIR / "train.csv"
    validation_path = OUTPUT_DIR / "validation.csv"
    test_path = OUTPUT_DIR / "test.csv"

    train.to_csv(
        train_path,
        index=False
    )

    validation.to_csv(
        validation_path,
        index=False
    )

    test.to_csv(
        test_path,
        index=False
    )

    print("\nFiles saved:")

    print(
        f"Train      → {train_path}"
    )

    print(
        f"Validation → {validation_path}"
    )

    print(
        f"Test       → {test_path}"
    )


# ============================================================
# DISPLAY SPLIT INFORMATION
# ============================================================

def print_split_information(
    train,
    validation,
    test
):

    print("\n" + "=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)

    print("\nTRAIN")
    print(
        f"Rows: {len(train)}"
    )
    print(
        f"Date: {train['date'].min().date()} "
        f"→ {train['date'].max().date()}"
    )

    print("\nVALIDATION")
    print(
        f"Rows: {len(validation)}"
    )
    print(
        f"Date: {validation['date'].min().date()} "
        f"→ {validation['date'].max().date()}"
    )

    print("\nTEST")
    print(
        f"Rows: {len(test)}"
    )
    print(
        f"Date: {test['date'].min().date()} "
        f"→ {test['date'].max().date()}"
    )


# ============================================================
# TARGET AVAILABILITY
# ============================================================

def check_targets(
    train,
    validation,
    test
):

    print("\n" + "=" * 60)
    print("TARGET AVAILABILITY")
    print("=" * 60)

    for name, dataset in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test)
    ]:

        print(f"\n{name}")

        for target in TARGETS:

            missing = dataset[target].isna().sum()

            available = dataset[target].notna().sum()

            print(
                f"{target}: "
                f"{available} available, "
                f"{missing} missing"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("PEARLS AQI PREDICTOR")
    print("TIME-BASED ML DATA PREPARATION")
    print("=" * 60)

    df = load_data()

    train, validation, test = create_split(df)

    print_split_information(
        train,
        validation,
        test
    )

    check_targets(
        train,
        validation,
        test
    )

    save_splits(
        train,
        validation,
        test
    )

    print("\n")
    print("=" * 60)
    print("SPLIT COMPLETE")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()