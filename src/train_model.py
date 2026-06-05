"""
Phase 3 — Model Training & Evaluation
=======================================
Trains 4 ML models, evaluates them, and saves the best one.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from src.utils import (
    MODELS_DIR, OUTPUTS_DIR, RANDOM_STATE,
    calculate_all_metrics, print_metrics_table, save_model_metadata,
)
from src.data_preprocessing import load_and_preprocess


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


# ─── Model Definitions ──────────────────────────────────────────────────────
def get_models():
    """Return a dict of model_name → model instance with sensible defaults."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
    }


# ─── Train & Evaluate ───────────────────────────────────────────────────────
def train_and_evaluate(X_train, X_test, y_train, y_test, feature_cols):
    """
    Train all models and evaluate them.

    Returns:
        results: dict of {model_name: {metrics_dict}}
        trained_models: dict of {model_name: fitted_model}
        predictions: dict of {model_name: y_pred_array}
    """
    print("━" * 60)
    print("  PHASE 3 — MODEL TRAINING & EVALUATION")
    print("━" * 60)

    models = get_models()
    results = {}
    trained_models = {}
    predictions = {}

    for name, model in models.items():
        print(f"\n── Training: {name} ──")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = calculate_all_metrics(y_test.values, y_pred)
        results[name] = metrics
        trained_models[name] = model
        predictions[name] = y_pred

        print(f"  MAE:  {metrics['MAE']:.2f}")
        print(f"  RMSE: {metrics['RMSE']:.2f}")
        print(f"  R²:   {metrics['R2']:.4f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")

    # Print comparison table
    print_metrics_table(results)

    return results, trained_models, predictions


# ─── Save Best Model ────────────────────────────────────────────────────────
def save_best_model(results, trained_models, feature_cols):
    """Identify and save the best model (lowest MAE)."""
    best_name = min(results, key=lambda k: results[k]["MAE"])
    best_model = trained_models[best_name]
    best_metrics = results[best_name]

    print(f"\n★ Best Model: {best_name}")
    print(f"  MAE={best_metrics['MAE']:.2f}, R²={best_metrics['R2']:.4f}")

    # Save model
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"  ✓ Model saved to: {model_path}")

    # Save all models
    for name, model in trained_models.items():
        safe_name = name.lower().replace(" ", "_")
        path = os.path.join(MODELS_DIR, f"{safe_name}.joblib")
        joblib.dump(model, path)

    # Save metadata
    save_model_metadata(best_name, best_metrics, feature_cols)

    # Save all results
    results_path = os.path.join(MODELS_DIR, "all_results.json")
    import json
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ All results saved to: {results_path}")

    return best_name, best_model


# ─── Visualization: Actual vs Predicted ──────────────────────────────────────
def plot_actual_vs_predicted(y_test, y_pred, model_name):
    """Scatter plot of actual vs predicted demand."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_test, y_pred, alpha=0.15, s=8, color=ACCENT[0])
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "--", color=ACCENT[1], linewidth=2, label="Perfect Prediction")

    ax.set_title(f"Actual vs Predicted Demand — {model_name}", fontweight="bold")
    ax.set_xlabel("Actual Demand")
    ax.set_ylabel("Predicted Demand")
    ax.legend(facecolor="#1a1c23", edgecolor="#2a2d35")
    ax.set_aspect("equal")

    path = os.path.join(OUTPUTS_DIR, "13_actual_vs_predicted.png")
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved: 13_actual_vs_predicted.png")


# ─── Visualization: Residual Distribution ───────────────────────────────────
def plot_residuals(y_test, y_pred, model_name):
    """Histogram of residuals."""
    residuals = y_test - y_pred

    fig, ax = plt.subplots()
    ax.hist(residuals, bins=50, color=ACCENT[2], alpha=0.8, edgecolor="#0e1117")
    ax.axvline(0, color=ACCENT[1], linestyle="--", linewidth=2)
    ax.set_title(f"Residual Distribution — {model_name}", fontweight="bold")
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Frequency")

    path = os.path.join(OUTPUTS_DIR, "14_residual_distribution.png")
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved: 14_residual_distribution.png")


# ─── Visualization: Feature Importance ───────────────────────────────────────
def plot_feature_importance(model, feature_cols, model_name, top_n=15):
    """Bar chart of top N feature importances."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        print("  ⚠ Model does not have feature_importances_ attribute, skipping.")
        return

    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True)
    top_features = feat_imp.tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [ACCENT[0] if v >= top_features.median() else ACCENT[4] for v in top_features.values]
    ax.barh(top_features.index, top_features.values, color=colors, edgecolor="#0e1117")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontweight="bold")
    ax.set_xlabel("Importance")

    path = os.path.join(OUTPUTS_DIR, "15_feature_importance.png")
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved: 15_feature_importance.png")


# ─── Visualization: Model Comparison ────────────────────────────────────────
def plot_model_comparison(results):
    """Bar chart comparing all models across metrics."""
    metrics_names = ["MAE", "RMSE", "R2", "MAPE"]
    model_names = list(results.keys())

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for idx, metric in enumerate(metrics_names):
        ax = axes[idx]
        values = [results[m][metric] for m in model_names]
        bars = ax.bar(range(len(model_names)), values,
                      color=[ACCENT[i] for i in range(len(model_names))],
                      edgecolor="#0e1117")
        ax.set_title(metric, fontweight="bold", fontsize=13)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([n.replace(" ", "\n") for n in model_names], fontsize=8)

        # Annotate values
        for bar, val in zip(bars, values):
            fmt = f"{val:.4f}" if metric == "R2" else f"{val:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt, ha="center", va="bottom", fontweight="bold",
                    color="#fafafa", fontsize=9)

    fig.suptitle("Model Performance Comparison", fontweight="bold", fontsize=15, y=1.02)
    fig.tight_layout()

    path = os.path.join(OUTPUTS_DIR, "16_model_comparison.png")
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved: 16_model_comparison.png")


# ─── Master Pipeline ────────────────────────────────────────────────────────
def run_training():
    """Run the full training pipeline."""
    # Load preprocessed data
    df, df_encoded, X_train, X_test, y_train, y_test, feature_cols, label_encoders = load_and_preprocess()

    # Train and evaluate
    results, trained_models, predictions = train_and_evaluate(
        X_train, X_test, y_train, y_test, feature_cols
    )

    # Save best model
    best_name, best_model = save_best_model(results, trained_models, feature_cols)

    # Generate training visualizations
    print(f"\n── Generating Training Visualizations ──")
    y_pred_best = predictions[best_name]

    plot_actual_vs_predicted(y_test, y_pred_best, best_name)
    plot_residuals(y_test.values, y_pred_best, best_name)
    plot_feature_importance(best_model, feature_cols, best_name)
    plot_model_comparison(results)

    print(f"\n{'━' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best Model: {best_name} (R²={results[best_name]['R2']:.4f})")
    print(f"{'━' * 60}\n")

    return results, best_name


# ─── CLI Entry Point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_training()
