# ============================================================
# PEARLS AQI PREDICTOR
# Phase 2 — Naive Baseline
# ============================================================

import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

SPLIT_DIR = Path("data/processed/splits")

OUTPUT_DIR = Path("reports/ml")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TARGETS = [
    "AQI_t+1",
    "AQI_t+2",
    "AQI_t+3"
]


# ============================================================
# LOAD DATA
# ============================================================

def load_split(filename):

    path = SPLIT_DIR / filename

    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df


# ============================================================
# NAIVE PREDICTION
# ============================================================

def create_naive_predictions(df):

    """
    Naive forecasting assumption:

        Future AQI = Current AQI

    Therefore:

        AQI_t+1 prediction = AQI
        AQI_t+2 prediction = AQI
        AQI_t+3 prediction = AQI
    """

    predictions = {}

    for target in TARGETS:

        predictions[target] = df["AQI"]

    return predictions


# ============================================================
# EVALUATION
# ============================================================

def evaluate_predictions(
    df,
    predictions,
    dataset_name
):

    results = []

    print("\n" + "=" * 60)
    print(f"{dataset_name.upper()} BASELINE RESULTS")
    print("=" * 60)

    for target in TARGETS:

        # Keep only rows where actual target exists
        mask = df[target].notna()

        actual = df.loc[
            mask,
            target
        ]

        predicted = predictions[target][mask]

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        )

        mae = mean_absolute_error(
            actual,
            predicted
        )

        r2 = r2_score(
            actual,
            predicted
        )

        results.append({
            "dataset": dataset_name,
            "target": target,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "n_samples": len(actual)
        })

        print(f"\n{target}")

        print(
            f"RMSE: {rmse:.4f}"
        )

        print(
            f"MAE:  {mae:.4f}"
        )

        print(
            f"R²:   {r2:.4f}"
        )

        print(
            f"Samples: {len(actual)}"
        )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("PEARLS AQI PREDICTOR")
    print("NAIVE BASELINE")
    print("=" * 60)

    validation = load_split(
        "validation.csv"
    )

    test = load_split(
        "test.csv"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_predictions = (
        create_naive_predictions(
            validation
        )
    )

    validation_results = evaluate_predictions(
        validation,
        validation_predictions,
        "validation"
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_predictions = (
        create_naive_predictions(
            test
        )
    )

    test_results = evaluate_predictions(
        test,
        test_predictions,
        "test"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = pd.DataFrame(
        validation_results +
        test_results
    )

    output_path = (
        OUTPUT_DIR /
        "baseline_results.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("BASELINE COMPLETE")
    print("=" * 60)

    print(
        f"\nResults saved to:\n{output_path}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()