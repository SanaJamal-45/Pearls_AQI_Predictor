import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

SPLIT_DIR = Path("data/processed/splits")
OUTPUT_DIR = Path("reports/ml")
MODEL_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "AQI", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
    "Temperature", "Humidity", "Precipitation",
    "day", "day_of_week", "month", "year", "day_of_year",
    "AQI_lag_1", "AQI_lag_2", "AQI_lag_3", "AQI_lag_7",
    "AQI_change_1d", "AQI_change_3d", "AQI_change_7d",
    "AQI_rolling_mean_3", "AQI_rolling_mean_7",
    "PM2.5_rolling_mean_3", "PM2.5_rolling_mean_7",
    "PM10_rolling_mean_3", "PM10_rolling_mean_7",
]
TARGETS = ["AQI_t+1", "AQI_t+2", "AQI_t+3"]

# Best params found via RandomizedSearchCV (60 iters, 3-fold CV) per horizon
BEST_PARAMS = {
    "AQI_t+1": dict(n_estimators=500, max_depth=2, learning_rate=0.05,
                     subsample=1.0, colsample_bytree=1.0,
                     reg_alpha=0.5, reg_lambda=5.0, min_child_weight=3),
    "AQI_t+2": dict(n_estimators=300, max_depth=2, learning_rate=0.01,
                     subsample=0.6, colsample_bytree=0.7,
                     reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5),
    "AQI_t+3": dict(n_estimators=300, max_depth=2, learning_rate=0.01,
                     subsample=0.6, colsample_bytree=0.7,
                     reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5),
}

train = pd.read_csv(SPLIT_DIR / "train.csv")
validation = pd.read_csv(SPLIT_DIR / "validation.csv")
test = pd.read_csv(SPLIT_DIR / "test.csv")

results = []
for target in TARGETS:
    train_c = train.dropna(subset=[target])
    val_c = validation.dropna(subset=[target])
    test_c = test.dropna(subset=[target])

    X_train, y_train = train_c[FEATURES], train_c[target]
    X_val, y_val = val_c[FEATURES], val_c[target]
    X_test, y_test = test_c[FEATURES], test_c[target]

    model = XGBRegressor(random_state=42, n_jobs=-1, **BEST_PARAMS[target])
    model.fit(X_train, y_train)

    for split_name, X_eval, y_eval in [("validation", X_val, y_val), ("test", X_test, y_test)]:
        preds = model.predict(X_eval)
        rmse = np.sqrt(mean_squared_error(y_eval, preds))
        mae = mean_absolute_error(y_eval, preds)
        r2 = r2_score(y_eval, preds)
        print(f"{target} [{split_name}] RMSE={rmse:.3f} MAE={mae:.3f} R2={r2:.3f}")
        results.append({"target": target, "split": split_name, "RMSE": rmse, "MAE": mae, "R2": r2})

    joblib.dump(model, MODEL_DIR / f"xgboost_tuned_{target.replace('+', 'plus')}.joblib")

pd.DataFrame(results).to_csv(OUTPUT_DIR / "xgboost_tuned_results.csv", index=False)