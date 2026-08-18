import pandas as pd
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# Paths
# ============================================================

SPLIT_DIR = Path("data/processed/splits")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("reports/ml")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


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
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n" + "=" * 60)
print("PEARLS AQI PREDICTOR")
print("PYTORCH NEURAL NETWORK — UPDATED FEATURES")
print("=" * 60)

print(f"\nUsing device: {device}")


# ============================================================
# Neural Network
# ============================================================

class AQINetwork(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):

        return self.network(x)


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
# Training function
# ============================================================

def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    target
):

    # --------------------------------------------------------
    # Scale features
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train_scaled,
        dtype=torch.float32
    ).to(device)

    y_train_tensor = torch.tensor(
        y_train.values,
        dtype=torch.float32
    ).reshape(-1, 1).to(device)

    X_val_tensor = torch.tensor(
        X_val_scaled,
        dtype=torch.float32
    ).to(device)

    y_val_tensor = torch.tensor(
        y_val.values,
        dtype=torch.float32
    ).reshape(-1, 1).to(device)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = AQINetwork(
        input_size=len(FEATURES)
    ).to(device)

    # --------------------------------------------------------
    # Loss and optimizer
    # --------------------------------------------------------

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    # --------------------------------------------------------
    # Training settings
    # --------------------------------------------------------

    max_epochs = 1000

    patience = 75

    best_val_loss = float("inf")

    patience_counter = 0

    best_state = None

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(1, max_epochs + 1):

        model.train()

        optimizer.zero_grad()

        train_predictions = model(
            X_train_tensor
        )

        train_loss = criterion(
            train_predictions,
            y_train_tensor
        )

        train_loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            val_predictions = model(
                X_val_tensor
            )

            val_loss = criterion(
                val_predictions,
                y_val_tensor
            )

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        if val_loss.item() < best_val_loss:

            best_val_loss = val_loss.item()

            best_state = {
                key: value.cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if epoch % 50 == 0:

            print(
                f"Epoch {epoch:4d} | "
                f"Train Loss: {train_loss.item():.4f} | "
                f"Validation Loss: {val_loss.item():.4f}"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if patience_counter >= patience:

            print(
                f"Early stopping at epoch {epoch}"
            )

            break

    # ========================================================
    # Restore best model
    # ========================================================

    model.load_state_dict(
        best_state
    )

    model.to(device)

    # ========================================================
    # Final validation prediction
    # ========================================================

    model.eval()

    with torch.no_grad():

        predictions = model(
            X_val_tensor
        ).cpu().numpy().flatten()

    # ========================================================
    # Metrics
    # ========================================================

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

    # ========================================================
    # Save model
    # ========================================================

    model_path = (
        MODEL_DIR /
        f"pytorch_updated_{target}.pth"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": len(FEATURES),
            "features": FEATURES,
            "target": target
        },
        model_path
    )

    # ========================================================
    # Print
    # ========================================================

    print(
        f"\n{target} Results:"
    )

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
        "Model saved to:"
    )

    print(model_path)

    return {
        "target": target,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "n_samples": len(y_val)
    }


# ============================================================
# Train all targets
# ============================================================

results = []


for target in TARGETS:

    print("\n" + "-" * 60)

    print(
        f"Training PyTorch model for {target}"
    )

    # --------------------------------------------------------
    # Remove missing target rows
    # --------------------------------------------------------

    train_clean = train.dropna(
        subset=[target]
    ).copy()

    validation_clean = validation.dropna(
        subset=[target]
    ).copy()

    # --------------------------------------------------------
    # Features and target
    # --------------------------------------------------------

    X_train = train_clean[FEATURES]

    y_train = train_clean[target]

    X_val = validation_clean[FEATURES]

    y_val = validation_clean[target]

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Validation samples: {len(X_val)}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    result = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        target
    )

    results.append(result)


# ============================================================
# Results
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n")
print("=" * 60)
print("PYTORCH VALIDATION RESULTS — UPDATED FEATURES")
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
    "pytorch_updated_validation_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


print("\nResults saved to:")
print(results_path)

print("\n")
print("=" * 60)
print("UPDATED PYTORCH TRAINING COMPLETE")
print("=" * 60)