"""
Utility functions for the Demand Forecasting project.
Shared helpers for metrics calculation, file I/O, and constants.
"""

import os
import json
import numpy as np
from datetime import datetime


# ─── Project Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "demand_forecasting.csv")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PROCESSED_DATA_PATH = os.path.join(OUTPUTS_DIR, "processed_data.csv")

# Create directories if they don't exist
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ─── Constants ───────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE_RATIO = 0.20  # Last 20% of dates for testing

# Target column
TARGET = "Demand"

# Columns to drop before training (identifiers, target, raw date)
DROP_COLS_FOR_TRAINING = ["Date", "Store ID", "Product ID", "Demand"]

# Category typo fix
CATEGORY_TYPO_MAP = {"Fzurniture": "Furniture"}

# Categorical columns for one-hot encoding
ONEHOT_COLS = ["Category", "Region", "Weather Condition", "Seasonality"]

# Columns to label-encode
LABEL_ENCODE_COLS = ["Store ID", "Product ID"]


# ─── Metrics ─────────────────────────────────────────────────────────────────
def mean_absolute_error(y_true, y_pred):
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true, y_pred):
    """Calculate R² (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error (MAPE)."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_all_metrics(y_true, y_pred):
    """
    Calculate all evaluation metrics and return as a dictionary.

    Returns:
        dict with keys: MAE, RMSE, R2, MAPE
    """
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(root_mean_squared_error(y_true, y_pred), 4),
        "R2": round(r2_score(y_true, y_pred), 4),
        "MAPE": round(mean_absolute_percentage_error(y_true, y_pred), 2),
    }


# ─── Model Metadata ─────────────────────────────────────────────────────────
def save_model_metadata(model_name, metrics, feature_names, filepath=None):
    """Save model training metadata to a JSON file."""
    if filepath is None:
        filepath = os.path.join(MODELS_DIR, "model_metadata.json")

    metadata = {
        "model_name": model_name,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "feature_count": len(feature_names),
        "features": list(feature_names),
    }

    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Metadata saved to {filepath}")
    return metadata


def load_model_metadata(filepath=None):
    """Load model training metadata from JSON."""
    if filepath is None:
        filepath = os.path.join(MODELS_DIR, "model_metadata.json")

    with open(filepath, "r") as f:
        return json.load(f)


# ─── Pretty Printing ────────────────────────────────────────────────────────
def print_metrics_table(results_dict):
    """
    Pretty-print a comparison table of model metrics.

    Args:
        results_dict: dict of {model_name: {MAE, RMSE, R2, MAPE}}
    """
    print("\n" + "=" * 70)
    print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'MAPE %':>8}")
    print("-" * 70)
    for name, metrics in results_dict.items():
        print(
            f"{name:<25} {metrics['MAE']:>8.2f} {metrics['RMSE']:>8.2f} "
            f"{metrics['R2']:>8.4f} {metrics['MAPE']:>7.2f}%"
        )
    print("=" * 70)
