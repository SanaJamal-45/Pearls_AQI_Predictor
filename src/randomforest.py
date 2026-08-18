import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# Paths
# ============================================================

SPLIT_DIR = Path("data/processed/splits")
OUTPUT_DIR = Path("reports/ml")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Features
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

    "AQI_change_1d",
    "AQI_change_3d",
    "AQI_change_7d",

    "AQI_rolling_mean_3",
    "AQI_rolling_mean_7",

    "PM2.5_rolling_mean_3",
    "PM2.5_rolling_mean_7",

    "PM10_rolling_mean_3",
    "PM10_rolling_mean_7"
]


TARGETS = [
    "AQI_t+1",
    "AQI_t+2",
    "AQI_t+3"
]


# ============================================================
# Load data
# ============================================================

train = pd.read_csv(
    SPLIT_DIR / "train.csv"
)

validation = pd.read_csv(
    SPLIT_DIR / "validation.csv"
)


# ============================================================
# Random Forest configuration
# ============================================================

RF_PARAMETERS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42,
    "n_jobs": -1
}


# ============================================================
# Training
# ============================================================

results = []

all_feature_importance = []


for target in TARGETS:

    print("\n" + "-" * 60)

    print(
        f"Training Random Forest for {target}"
    )

    # --------------------------------------------------------
    # Remove rows where target is missing
    # --------------------------------------------------------

    train_clean = train.dropna(
        subset=[target]
    ).copy()

    validation_clean = validation.dropna(
        subset=[target]
    ).copy()

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X_train = train_clean[FEATURES]
    y_train = train_clean[target]

    X_val = validation_clean[FEATURES]
    y_val = validation_clean[target]

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Validation samples: {len(X_val)}"
    )

    # --------------------------------------------------------
    # Create Random Forest
    # --------------------------------------------------------

    model = RandomForestRegressor(
        **RF_PARAMETERS
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = model.predict(
        X_val
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(
            y_val,
            predictions
        )
    )

    mae = mean_absolute_error(
        y_val,
        predictions
    )

    r2 = r2_score(
        y_val,
        predictions
    )

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print(
        f"\nRMSE: {rmse:.4f}"
    )

    print(
        f"MAE:  {mae:.4f}"
    )

    print(
        f"R²:   {r2:.4f}"
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance_df = pd.DataFrame({
        "target": target,
        "feature": FEATURES,
        "importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    )

    all_feature_importance.append(
        importance_df
    )

    print("\nTop 10 features:")

    print(
        importance_df[
            ["feature", "importance"]
        ]
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({
        "target": target,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "n_samples": len(y_val)
    })


# ============================================================
# Results DataFrame
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n")
print("=" * 60)
print("RANDOM FOREST VALIDATION RESULTS — UPDATED FEATURES")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# Save validation results
# ============================================================

results_path = (
    OUTPUT_DIR /
    "random_forest_updated_validation_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# Save feature importance
# ============================================================

feature_importance_df = pd.concat(
    all_feature_importance,
    ignore_index=True
)

importance_path = (
    OUTPUT_DIR /
    "random_forest_updated_feature_importance.csv"
)

feature_importance_df.to_csv(
    importance_path,
    index=False
)


# ============================================================
# Final messages
# ============================================================

print("\nResults saved to:")
print(results_path)

print("\nFeature importance saved to:")
print(importance_path)

print("\n")
print("=" * 60)
print("UPDATED RANDOM FOREST TRAINING COMPLETE")
print("=" * 60)