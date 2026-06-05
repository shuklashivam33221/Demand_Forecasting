"""
Phase 2 — Exploratory Data Analysis
=====================================
Generates 12 professional visualizations and saves them as PNGs.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import OUTPUTS_DIR
from src.data_preprocessing import load_raw_data, clean_data, engineer_features

# ─── Style Configuration ────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="viridis")
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#1a1c23",
    "text.color": "#fafafa",
    "axes.labelcolor": "#fafafa",
    "xtick.color": "#cccccc",
    "ytick.color": "#cccccc",
    "grid.color": "#2a2d35",
})

ACCENT_COLORS = ["#00d4aa", "#ff6b6b", "#4ecdc4", "#ffe66d", "#a78bfa", "#f97316"]


def save_fig(fig, name):
    """Save figure and close."""
    path = os.path.join(OUTPUTS_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"  ✓ Saved: {name}.png")
    return path


# ─── 1. Demand Distribution ─────────────────────────────────────────────────
def plot_demand_distribution(df):
    fig, ax = plt.subplots()
    ax.hist(df["Demand"], bins=50, color=ACCENT_COLORS[0], alpha=0.8, edgecolor="#0e1117")
    ax.axvline(df["Demand"].mean(), color=ACCENT_COLORS[1], linestyle="--", linewidth=2,
               label=f'Mean: {df["Demand"].mean():.1f}')
    ax.axvline(df["Demand"].median(), color=ACCENT_COLORS[3], linestyle="--", linewidth=2,
               label=f'Median: {df["Demand"].median():.1f}')
    ax.set_title("Distribution of Demand", fontweight="bold")
    ax.set_xlabel("Demand")
    ax.set_ylabel("Frequency")
    ax.legend(facecolor="#1a1c23", edgecolor="#2a2d35")
    return save_fig(fig, "01_demand_distribution")


# ─── 2. Demand Over Time ────────────────────────────────────────────────────
def plot_demand_over_time(df):
    daily_demand = df.groupby("Date")["Demand"].mean().reset_index()
    # Rolling average for smoothing
    daily_demand["rolling_7d"] = daily_demand["Demand"].rolling(7).mean()
    daily_demand["rolling_30d"] = daily_demand["Demand"].rolling(30).mean()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(daily_demand["Date"], daily_demand["Demand"], alpha=0.3, color=ACCENT_COLORS[0], linewidth=0.8, label="Daily")
    ax.plot(daily_demand["Date"], daily_demand["rolling_7d"], color=ACCENT_COLORS[2], linewidth=1.5, label="7-Day MA")
    ax.plot(daily_demand["Date"], daily_demand["rolling_30d"], color=ACCENT_COLORS[1], linewidth=2, label="30-Day MA")
    ax.set_title("Average Daily Demand Over Time", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Demand")
    ax.legend(facecolor="#1a1c23", edgecolor="#2a2d35")
    return save_fig(fig, "02_demand_over_time")


# ─── 3. Demand by Category ──────────────────────────────────────────────────
def plot_demand_by_category(df):
    fig, ax = plt.subplots()
    order = df.groupby("Category")["Demand"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="Category", y="Demand", order=order, ax=ax,
                palette=ACCENT_COLORS[:len(order)], fliersize=2)
    ax.set_title("Demand Distribution by Category", fontweight="bold")
    ax.set_xlabel("Category")
    ax.set_ylabel("Demand")
    return save_fig(fig, "03_demand_by_category")


# ─── 4. Demand by Region ────────────────────────────────────────────────────
def plot_demand_by_region(df):
    fig, ax = plt.subplots()
    order = df.groupby("Region")["Demand"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="Region", y="Demand", order=order, ax=ax,
                palette=ACCENT_COLORS[:len(order)], fliersize=2)
    ax.set_title("Demand Distribution by Region", fontweight="bold")
    ax.set_xlabel("Region")
    ax.set_ylabel("Demand")
    return save_fig(fig, "04_demand_by_region")


# ─── 5. Demand by Season ────────────────────────────────────────────────────
def plot_demand_by_season(df):
    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    season_means = df.groupby("Seasonality")["Demand"].mean().reindex(season_order)

    fig, ax = plt.subplots()
    bars = ax.bar(season_means.index, season_means.values, color=ACCENT_COLORS[:4], edgecolor="#0e1117", linewidth=1.5)
    for bar, val in zip(bars, season_means.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}", ha="center", va="bottom", fontweight="bold", color="#fafafa")
    ax.set_title("Average Demand by Season", fontweight="bold")
    ax.set_xlabel("Season")
    ax.set_ylabel("Average Demand")
    return save_fig(fig, "05_demand_by_season")


# ─── 6. Demand by Weather ───────────────────────────────────────────────────
def plot_demand_by_weather(df):
    fig, ax = plt.subplots()
    order = df.groupby("Weather Condition")["Demand"].median().sort_values(ascending=False).index
    sns.violinplot(data=df, x="Weather Condition", y="Demand", order=order, ax=ax,
                   palette=ACCENT_COLORS[:len(order)], inner="quartile")
    ax.set_title("Demand Distribution by Weather Condition", fontweight="bold")
    ax.set_xlabel("Weather Condition")
    ax.set_ylabel("Demand")
    return save_fig(fig, "06_demand_by_weather")


# ─── 7. Promotion Impact ────────────────────────────────────────────────────
def plot_promotion_impact(df):
    fig, ax = plt.subplots()
    promo_labels = {0: "No Promotion", 1: "Promotion Active"}
    df_plot = df.copy()
    df_plot["Promo Status"] = df_plot["Promotion"].map(promo_labels)

    sns.violinplot(data=df_plot, x="Promo Status", y="Demand", ax=ax,
                   palette=[ACCENT_COLORS[1], ACCENT_COLORS[0]], inner="quartile")

    # Add mean annotations
    for i, promo in enumerate([0, 1]):
        mean_val = df[df["Promotion"] == promo]["Demand"].mean()
        ax.text(i, mean_val + 5, f"μ={mean_val:.1f}", ha="center", fontweight="bold",
                color=ACCENT_COLORS[3], fontsize=12)

    ax.set_title("Impact of Promotions on Demand", fontweight="bold")
    ax.set_ylabel("Demand")
    return save_fig(fig, "07_promotion_impact")


# ─── 8. Discount vs Demand ──────────────────────────────────────────────────
def plot_discount_vs_demand(df):
    fig, ax = plt.subplots()
    discount_means = df.groupby("Discount")["Demand"].mean().reset_index()
    ax.bar(discount_means["Discount"], discount_means["Demand"],
           color=ACCENT_COLORS[0], edgecolor="#0e1117", alpha=0.85)
    ax.set_title("Average Demand by Discount Level", fontweight="bold")
    ax.set_xlabel("Discount (%)")
    ax.set_ylabel("Average Demand")
    return save_fig(fig, "08_discount_vs_demand")


# ─── 9. Price vs Demand ─────────────────────────────────────────────────────
def plot_price_vs_demand(df):
    fig, ax = plt.subplots()
    categories = df["Category"].unique()
    colors = {cat: ACCENT_COLORS[i % len(ACCENT_COLORS)] for i, cat in enumerate(categories)}

    for cat in categories:
        subset = df[df["Category"] == cat].sample(min(500, len(df[df["Category"] == cat])), random_state=42)
        ax.scatter(subset["Price"], subset["Demand"], alpha=0.3, s=10,
                   color=colors[cat], label=cat)

    ax.set_title("Price vs Demand (by Category)", fontweight="bold")
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Demand")
    ax.legend(facecolor="#1a1c23", edgecolor="#2a2d35", markerscale=3)
    return save_fig(fig, "09_price_vs_demand")


# ─── 10. Correlation Heatmap ────────────────────────────────────────────────
def plot_correlation_heatmap(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5, annot_kws={"size": 7},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontweight="bold", pad=20)
    return save_fig(fig, "10_correlation_heatmap")


# ─── 11. Store-wise Monthly Trend ───────────────────────────────────────────
def plot_store_monthly_trend(df):
    df_temp = df.copy()
    df_temp["YearMonth"] = df_temp["Date"].dt.to_period("M")
    monthly = df_temp.groupby(["YearMonth", "Store ID"])["Demand"].mean().reset_index()
    monthly["YearMonth"] = monthly["YearMonth"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(14, 6))
    stores = sorted(monthly["Store ID"].unique())
    for i, store in enumerate(stores):
        store_data = monthly[monthly["Store ID"] == store]
        ax.plot(store_data["YearMonth"], store_data["Demand"],
                marker="o", markersize=3, linewidth=1.5,
                color=ACCENT_COLORS[i % len(ACCENT_COLORS)], label=store)

    ax.set_title("Monthly Average Demand by Store", fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Demand")
    ax.legend(facecolor="#1a1c23", edgecolor="#2a2d35")
    return save_fig(fig, "11_store_monthly_trend")


# ─── 12. Top & Bottom Products ──────────────────────────────────────────────
def plot_top_bottom_products(df):
    product_demand = df.groupby("Product ID")["Demand"].mean().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = [ACCENT_COLORS[0] if v >= product_demand.median() else ACCENT_COLORS[1]
              for v in product_demand.values]
    ax.barh(product_demand.index, product_demand.values, color=colors, edgecolor="#0e1117")
    ax.axvline(product_demand.mean(), color=ACCENT_COLORS[3], linestyle="--", linewidth=2,
               label=f"Mean: {product_demand.mean():.1f}")
    ax.set_title("Average Demand by Product", fontweight="bold")
    ax.set_xlabel("Average Demand")
    ax.set_ylabel("Product ID")
    ax.legend(facecolor="#1a1c23", edgecolor="#2a2d35")
    return save_fig(fig, "12_product_demand_ranking")


# ─── Run All EDA ────────────────────────────────────────────────────────────
def run_eda():
    """Run the complete EDA pipeline."""
    print("━" * 60)
    print("  PHASE 2 — EXPLORATORY DATA ANALYSIS")
    print("━" * 60)

    # Load and clean data (without encoding, to keep original categories)
    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)

    print(f"\n── Generating Visualizations ──")

    plot_demand_distribution(df)
    plot_demand_over_time(df)
    plot_demand_by_category(df)
    plot_demand_by_region(df)
    plot_demand_by_season(df)
    plot_demand_by_weather(df)
    plot_promotion_impact(df)
    plot_discount_vs_demand(df)
    plot_price_vs_demand(df)
    plot_correlation_heatmap(df)
    plot_store_monthly_trend(df)
    plot_top_bottom_products(df)

    print(f"\n{'━' * 60}")
    print(f"  EDA COMPLETE — 12 visualizations saved to outputs/")
    print(f"{'━' * 60}\n")


# ─── CLI Entry Point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_eda()
