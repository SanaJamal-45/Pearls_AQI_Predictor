import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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
# Model
# ============================================================

def create_model():

    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "ridge",
            Ridge(alpha=1.0)
        )
    ])


# ============================================================
# Training
# ============================================================

results = []


for target in TARGETS:

    print("\n" + "-" * 60)

    print(
        f"Training Ridge for {target}"
    )

    # --------------------------------------------------------
    # Remove rows with missing target
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

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_model()

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
    # Print
    # --------------------------------------------------------

    print(
        f"RMSE: {rmse:.4f}"
    )

    print(
        f"MAE:  {mae:.4f}"
    )

    print(
        f"R²:   {r2:.4f}"
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
# Results
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n")
print("=" * 60)
print("RIDGE VALIDATION RESULTS — UPDATED FEATURES")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# Save
# ============================================================

output_path = (
    OUTPUT_DIR /
    "ridge_updated_validation_results.csv"
)

results_df.to_csv(
    output_path,
    index=False
)


print("\nResults saved to:")
print(output_path)

print("\n")
print("=" * 60)
print("UPDATED RIDGE TRAINING COMPLETE")
print("=" * 60)