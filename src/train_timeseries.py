"""
Phase 3B — Time-Series & Deep Learning Models
================================================
Trains ARIMA, Prophet, XGBoost (lag-based), and LSTM on aggregated daily demand.
Compares all models and saves results + visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    DATA_PATH, MODELS_DIR, OUTPUTS_DIR, RANDOM_STATE,
    calculate_all_metrics, print_metrics_table,
)

# ─── Style ───────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#1a1c23",
    "text.color": "#fafafa",
    "axes.labelcolor": "#fafafa",
    "xtick.color": "#cccccc",
    "ytick.color": "#cccccc",
    "grid.color": "#2a2d35",
})
ACCENT = ["#00d4aa", "#ff6b6b", "#4ecdc4", "#ffe66d", "#a78bfa", "#f97316"]


# ─── 1. Data Preparation ────────────────────────────────────────────────────
def prepare_timeseries_data(filepath=DATA_PATH):
    """Aggregate raw data into daily total demand for time-series modeling."""
    print("━" * 60)
    print("  PHASE 3B — TIME-SERIES & DEEP LEARNING MODELS")
    print("━" * 60)

    df = pd.read_csv(filepath, parse_dates=["Date"])
    df["Category"] = df["Category"].replace("Fzurniture", "Furniture")

    # Aggregate to daily total demand
    daily = df.groupby("Date").agg({"Demand": "sum"}).reset_index()
    daily = daily.sort_values("Date").reset_index(drop=True)
    daily.columns = ["Date", "Demand"]

    print(f"\n✓ Aggregated to daily demand: {len(daily)} days")
    print(f"  Date range: {daily['Date'].min().date()} → {daily['Date'].max().date()}")
    print(f"  Daily demand: mean={daily['Demand'].mean():.0f}, std={daily['Demand'].std():.0f}")

    # Time-based split (same 80/20 as tabular)
    split_idx = int(len(daily) * 0.80)
    train = daily.iloc[:split_idx].copy()
    test = daily.iloc[split_idx:].copy()

    print(f"  Train: {len(train)} days | Test: {len(test)} days")
    print(f"  Split date: {test['Date'].iloc[0].date()}")

    return daily, train, test


# ─── 2. ARIMA ────────────────────────────────────────────────────────────────
def train_arima(train, test):
    """Train ARIMA model on daily demand."""
    from statsmodels.tsa.arima.model import ARIMA

    print(f"\n── Training: ARIMA(5,1,2) ──")

    train_values = train["Demand"].values

    model = ARIMA(train_values, order=(5, 1, 2))
    fitted = model.fit()

    forecast = fitted.forecast(steps=len(test))
    y_true = test["Demand"].values
    metrics = calculate_all_metrics(y_true, forecast)

    print(f"  MAE:  {metrics['MAE']:.2f}")
    print(f"  RMSE: {metrics['RMSE']:.2f}")
    print(f"  R²:   {metrics['R2']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")

    return forecast, metrics


# ─── 3. Prophet ──────────────────────────────────────────────────────────────
def train_prophet(train, test):
    """Train Facebook Prophet on daily demand."""
    from prophet import Prophet

    print(f"\n── Training: Prophet ──")

    prophet_train = train[["Date", "Demand"]].rename(columns={"Date": "ds", "Demand": "y"})

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    model.fit(prophet_train)

    future = model.make_future_dataframe(periods=len(test))
    forecast_df = model.predict(future)

    forecast = forecast_df["yhat"].iloc[-len(test):].values
    y_true = test["Demand"].values
    metrics = calculate_all_metrics(y_true, forecast)

    print(f"  MAE:  {metrics['MAE']:.2f}")
    print(f"  RMSE: {metrics['RMSE']:.2f}")
    print(f"  R²:   {metrics['R2']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")

    return forecast, metrics


# ─── 4. XGBoost with Lag Features ────────────────────────────────────────────
def train_xgboost_ts(train, test, daily):
    """Train XGBoost using lag and rolling window features on daily demand."""
    from xgboost import XGBRegressor

    print(f"\n── Training: XGBoost (Lag Features) ──")

    df = daily.copy()

    # Create lag features
    for lag in [1, 3, 7, 14, 21, 30]:
        df[f"lag_{lag}"] = df["Demand"].shift(lag)

    # Rolling statistics
    df["rolling_7_mean"] = df["Demand"].shift(1).rolling(7).mean()
    df["rolling_7_std"] = df["Demand"].shift(1).rolling(7).std()
    df["rolling_30_mean"] = df["Demand"].shift(1).rolling(30).mean()
    df["rolling_30_std"] = df["Demand"].shift(1).rolling(30).std()

    # Date features
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["day_of_month"] = df["Date"].dt.day

    # Drop rows with NaN from lagging
    df = df.dropna().reset_index(drop=True)

    # Re-split after creating features
    split_date = test["Date"].iloc[0]
    train_ts = df[df["Date"] < split_date]
    test_ts = df[df["Date"] >= split_date]

    feature_cols = [c for c in train_ts.columns if c not in ["Date", "Demand"]]

    X_train = train_ts[feature_cols]
    y_train = train_ts["Demand"]
    X_test = test_ts[feature_cols]
    y_test = test_ts["Demand"]

    model = XGBRegressor(
        n_estimators=300, max_depth=7, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    )
    model.fit(X_train, y_train)
    forecast = model.predict(X_test)

    y_true = y_test.values
    metrics = calculate_all_metrics(y_true, forecast)

    print(f"  MAE:  {metrics['MAE']:.2f}")
    print(f"  RMSE: {metrics['RMSE']:.2f}")
    print(f"  R²:   {metrics['R2']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")

    # Save this model
    joblib.dump(model, os.path.join(MODELS_DIR, "xgboost_timeseries.joblib"))

    return forecast, metrics, test_ts["Date"].values


# ─── 5. LSTM ─────────────────────────────────────────────────────────────────
def train_lstm(train, test, lookback=30, epochs=50, batch_size=32):
    """Train LSTM neural network on daily demand sequences."""
    print(f"\n── Training: LSTM (lookback={lookback}, epochs={epochs}) ──")

    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        print("  ⚠ TensorFlow not installed. Skipping LSTM.")
        return None, None

    # Normalize
    scaler = MinMaxScaler()
    all_demand = pd.concat([train["Demand"], test["Demand"]]).values.reshape(-1, 1)
    scaler.fit(train["Demand"].values.reshape(-1, 1))

    train_scaled = scaler.transform(train["Demand"].values.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test["Demand"].values.reshape(-1, 1)).flatten()

    # Create sequences from full data
    full_scaled = np.concatenate([train_scaled, test_scaled])
    train_len = len(train_scaled)

    def create_sequences(data, lookback):
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i - lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    X_all, y_all = create_sequences(full_scaled, lookback)

    # Split: training sequences end where test data begins
    split_point = train_len - lookback
    X_train = X_all[:split_point]
    y_train = y_all[:split_point]
    X_test = X_all[split_point:]
    y_test = y_all[split_point:]

    # Reshape for LSTM: [samples, timesteps, features]
    X_train = X_train.reshape(-1, lookback, 1)
    X_test = X_test.reshape(-1, lookback, 1)

    # Build model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[early_stop],
        verbose=0,
    )

    # Predict and inverse-transform
    pred_scaled = model.predict(X_test, verbose=0).flatten()
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    metrics = calculate_all_metrics(actual, pred)

    print(f"  MAE:  {metrics['MAE']:.2f}")
    print(f"  RMSE: {metrics['RMSE']:.2f}")
    print(f"  R²:   {metrics['R2']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  Stopped at epoch: {len(history.history['loss'])}")

    # Save model
    model.save(os.path.join(MODELS_DIR, "lstm_model.keras"))

    return pred, metrics


# ─── Visualizations ──────────────────────────────────────────────────────────
def plot_ts_forecast_comparison(test, predictions, model_names):
    """Plot actual vs all model forecasts."""
    fig, ax = plt.subplots(figsize=(14, 6))

    dates = test["Date"].values[:len(list(predictions.values())[0])]
    actual = test["Demand"].values[:len(dates)]

    ax.plot(dates, actual, color="#ffffff", linewidth=2, label="Actual", alpha=0.9)

    for i, name in enumerate(model_names):
        pred = predictions[name][:len(dates)]
        ax.plot(dates, pred, color=ACCENT[i % len(ACCENT)], linewidth=1.5,
                label=name, alpha=0.8, linestyle="--")

    ax.set_title("Time-Series Forecast Comparison", fontweight="bold", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Total Demand")
    ax.legend(facecolor="#1a1c23", edgecolor="#2a2d35", fontsize=9)

    path = os.path.join(OUTPUTS_DIR, "17_ts_forecast_comparison.png")
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved: 17_ts_forecast_comparison.png")


def plot_ts_model_comparison(results):
    """Bar chart comparing time-series models."""
    metrics_names = ["MAE", "RMSE", "R2", "MAPE"]
    model_names = list(results.keys())

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for idx, metric in enumerate(metrics_names):
        ax = axes[idx]
        values = [results[m][metric] for m in model_names]
        bars = ax.bar(range(len(model_names)), values,
                      color=[ACCENT[i % len(ACCENT)] for i in range(len(model_names))],
                      edgecolor="#0e1117")
        ax.set_title(metric, fontweight="bold", fontsize=13)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([n.replace(" ", "\n") for n in model_names], fontsize=8)

        for bar, val in zip(bars, values):
            fmt = f"{val:.4f}" if metric == "R2" else f"{val:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt, ha="center", va="bottom", fontweight="bold",
                    color="#fafafa", fontsize=9)

    fig.suptitle("Time-Series Model Comparison", fontweight="bold", fontsize=15, y=1.02)
    fig.tight_layout()

    path = os.path.join(OUTPUTS_DIR, "18_ts_model_comparison.png")
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved: 18_ts_model_comparison.png")


def plot_ts_residuals(test, predictions, results):
    """Residual distribution for each time-series model."""
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    actual = test["Demand"].values

    for i, (name, pred) in enumerate(predictions.items()):
        ax = axes[i]
        length = min(len(actual), len(pred))
        residuals = actual[:length] - pred[:length]
        ax.hist(residuals, bins=30, color=ACCENT[i % len(ACCENT)], alpha=0.8, edgecolor="#0e1117")
        ax.axvline(0, color="#ff6b6b", linestyle="--", linewidth=2)
        ax.set_title(f"{name}\n(MAE={results[name]['MAE']:.1f})", fontweight="bold", fontsize=10)
        ax.set_xlabel("Residual")

    fig.suptitle("Residual Distribution — Time-Series Models", fontweight="bold", fontsize=13, y=1.05)
    fig.tight_layout()

    path = os.path.join(OUTPUTS_DIR, "19_ts_residuals.png")
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved: 19_ts_residuals.png")


# ─── Master Pipeline ────────────────────────────────────────────────────────
def run_timeseries_training():
    """Run the full time-series training pipeline."""
    daily, train, test = prepare_timeseries_data()

    results = {}
    predictions = {}

    # 1. ARIMA
    try:
        pred, metrics = train_arima(train, test)
        results["ARIMA"] = metrics
        predictions["ARIMA"] = pred
    except Exception as e:
        print(f"  ⚠ ARIMA failed: {e}")

    # 2. Prophet
    try:
        pred, metrics = train_prophet(train, test)
        results["Prophet"] = metrics
        predictions["Prophet"] = pred
    except Exception as e:
        print(f"  ⚠ Prophet failed: {e}")

    # 3. XGBoost (Lag Features)
    try:
        pred, metrics, xgb_dates = train_xgboost_ts(train, test, daily)
        results["XGBoost (Lags)"] = metrics
        predictions["XGBoost (Lags)"] = pred
    except Exception as e:
        print(f"  ⚠ XGBoost (Lags) failed: {e}")

    # 4. LSTM
    try:
        pred, metrics = train_lstm(train, test)
        if pred is not None:
            results["LSTM"] = metrics
            predictions["LSTM"] = pred
    except Exception as e:
        print(f"  ⚠ LSTM failed: {e}")

    # Print comparison
    if results:
        print_metrics_table(results)

    # Save results
    ts_results_path = os.path.join(MODELS_DIR, "ts_results.json")
    with open(ts_results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Results saved to: {ts_results_path}")

    # Generate visualizations
    print(f"\n── Generating Time-Series Visualizations ──")
    if predictions:
        plot_ts_forecast_comparison(test, predictions, list(predictions.keys()))
        plot_ts_model_comparison(results)
        plot_ts_residuals(test, predictions, results)

    print(f"\n{'━' * 60}")
    print(f"  TIME-SERIES TRAINING COMPLETE")
    if results:
        best_name = min(results, key=lambda k: results[k]["MAE"])
        print(f"  Best TS Model: {best_name} (R²={results[best_name]['R2']:.4f})")
    print(f"{'━' * 60}\n")

    return results


# ─── CLI Entry Point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_timeseries_training()
