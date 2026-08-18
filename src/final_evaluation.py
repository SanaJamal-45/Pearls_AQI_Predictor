import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import joblib


# ============================================================
# Paths
# ============================================================

SPLIT_DIR = Path("data/processed/splits")
OUTPUT_DIR = Path("reports/ml")
MODEL_DIR = Path("models")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


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
# Load datasets
# ============================================================

train = pd.read_csv(
    SPLIT_DIR / "train.csv"
)

validation = pd.read_csv(
    SPLIT_DIR / "validation.csv"
)

test = pd.read_csv(
    SPLIT_DIR / "test.csv"
)


# ============================================================
# Combine train + validation
# ============================================================

train_validation = pd.concat(
    [
        train,
        validation
    ],
    ignore_index=True
)


# ============================================================
# Display dataset periods
# ============================================================

print("\n" + "=" * 60)
print("PEARLS AQI PREDICTOR")
print("FINAL RIDGE TEST EVALUATION — UPDATED FEATURES")
print("=" * 60)

print("\nSelected model: Ridge Regression")

print("\nDataset periods:")

print(
    f"Train:      {train['date'].min()} → "
    f"{train['date'].max()}"
)

print(
    f"Validation: {validation['date'].min()} → "
    f"{validation['date'].max()}"
)

print(
    f"Test:       {test['date'].min()} → "
    f"{test['date'].max()}"
)

print(
    f"\nFeatures used: {len(FEATURES)}"
)


# ============================================================
# Train and evaluate each horizon
# ============================================================

results = []
all_predictions = []


for target in TARGETS:

    print("\n" + "-" * 60)

    print(
        f"Final evaluation for {target}"
    )

    # --------------------------------------------------------
    # Remove rows with missing target
    # --------------------------------------------------------

    train_clean = train_validation.dropna(
        subset=[target]
    ).copy()

    test_clean = test.dropna(
        subset=[target]
    ).copy()

    # --------------------------------------------------------
    # Features and target
    # --------------------------------------------------------

    X_train = train_clean[FEATURES]
    y_train = train_clean[target]

    X_test = test_clean[FEATURES]
    y_test = test_clean[target]

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Test samples: {len(X_test)}"
    )

    # --------------------------------------------------------
    # Ridge model
    # --------------------------------------------------------

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "ridge",
            Ridge(alpha=1.0)
        )
    ])

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
        X_test
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # Print metrics
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
    # Save model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR /
        f"ridge_final_{target}.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        "Model saved to:"
    )

    print(model_path)

    # --------------------------------------------------------
    # Store metrics
    # --------------------------------------------------------

    results.append({
        "target": target,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "n_samples": len(y_test)
    })

    # --------------------------------------------------------
    # Store predictions
    # --------------------------------------------------------

    prediction_df = pd.DataFrame({
        "date": test_clean["date"].values,
        "target": target,
        "actual": y_test.values,
        "predicted": predictions,
        "error": y_test.values - predictions,
        "absolute_error": np.abs(
            y_test.values - predictions
        )
    })

    all_predictions.append(
        prediction_df
    )


# ============================================================
# Results DataFrame
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# Print final results
# ============================================================

print("\n")
print("=" * 60)
print("FINAL TEST RESULTS — UPDATED FEATURES")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# Save results
# ============================================================

results_path = (
    OUTPUT_DIR /
    "final_test_results_updated.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# Save predictions
# ============================================================

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

predictions_path = (
    OUTPUT_DIR /
    "final_test_predictions_updated.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# Print saved files
# ============================================================

print("\nResults saved to:")
print(results_path)

print("\nPredictions saved to:")
print(predictions_path)


# ============================================================
# Final summary
# ============================================================

print("\n")
print("=" * 60)
print("FINAL EVALUATION COMPLETE")
print("=" * 60)

print("\nModel selected:")
print("Ridge Regression")

print("\nTraining strategy:")
print("Train + Validation → Final model")

print("\nTest strategy:")
print("Completely held-out test period")

print("\nNo test data was used during model selection.")