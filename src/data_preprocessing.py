"""
Phase 1 — Data Loading & Preprocessing
========================================
Loads the raw CSV, cleans data, engineers features, and encodes categoricals.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    DATA_PATH,
    PROCESSED_DATA_PATH,
    CATEGORY_TYPO_MAP,
    ONEHOT_COLS,
    LABEL_ENCODE_COLS,
    TARGET,
    DROP_COLS_FOR_TRAINING,
    TEST_SIZE_RATIO,
)


# ─── 1.1  Load & Initial Inspection ─────────────────────────────────────────
def load_raw_data(filepath=DATA_PATH):
    """Load the raw CSV with proper date parsing."""
    print("━" * 60)
    print("  PHASE 1 — DATA PREPROCESSING")
    print("━" * 60)

    df = pd.read_csv(filepath, parse_dates=["Date"])
    print(f"\n✓ Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"  Columns: {list(df.columns)}")
    return df


# ─── 1.2  Data Cleaning ─────────────────────────────────────────────────────
def clean_data(df):
    """Fix typos, handle nulls, remove duplicates."""
    print("\n── Cleaning ──")

    # Fix category typo: Fzurniture → Furniture
    for wrong, correct in CATEGORY_TYPO_MAP.items():
        count = (df["Category"] == wrong).sum()
        if count > 0:
            df["Category"] = df["Category"].replace(wrong, correct)
            print(f"  ✓ Fixed typo: '{wrong}' → '{correct}' ({count} rows)")

    # Check for nulls
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        print(f"  ⚠ Found {total_nulls} null values:")
        print(null_counts[null_counts > 0])
        df = df.dropna()
        print(f"  ✓ Dropped rows with nulls. New shape: {df.shape}")
    else:
        print("  ✓ No null values found")

    # Remove duplicates
    dupes = df.duplicated().sum()
    if dupes > 0:
        df = df.drop_duplicates()
        print(f"  ✓ Removed {dupes} duplicate rows. New shape: {df.shape}")
    else:
        print("  ✓ No duplicate rows found")

    # Validate binary columns
    for col in ["Promotion", "Epidemic"]:
        unique_vals = sorted(df[col].unique())
        print(f"  ✓ {col} values: {unique_vals}")

    return df


# ─── 1.3  Feature Engineering ────────────────────────────────────────────────
def engineer_features(df):
    """Create derived features from the raw data."""
    print("\n── Feature Engineering ──")

    # Date-based features
    df["day_of_week"] = df["Date"].dt.dayofweek  # 0=Mon, 6=Sun
    df["day_of_month"] = df["Date"].dt.day
    df["month"] = df["Date"].dt.month
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["Date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)

    # Price-based features
    df["price_discount_ratio"] = df["Price"] * (1 - df["Discount"] / 100)
    df["price_competitor_diff"] = df["Price"] - df["Competitor Pricing"]
    df["price_competitor_ratio"] = df["Price"] / (df["Competitor Pricing"] + 1e-6)

    # Inventory-based features
    df["inventory_sold_ratio"] = df["Inventory Level"] / (df["Units Sold"] + 1)
    df["sold_ordered_ratio"] = df["Units Sold"] / (df["Units Ordered"] + 1)

    feature_names = [
        "day_of_week", "day_of_month", "month", "week_of_year",
        "is_weekend", "is_month_start", "is_month_end",
        "price_discount_ratio", "price_competitor_diff", "price_competitor_ratio",
        "inventory_sold_ratio", "sold_ordered_ratio",
    ]
    print(f"  ✓ Created {len(feature_names)} new features: {feature_names}")

    return df


# ─── 1.4  Encoding ──────────────────────────────────────────────────────────
def encode_features(df):
    """Label-encode IDs, one-hot-encode categoricals."""
    print("\n── Encoding ──")

    # Label encode Store ID and Product ID
    label_encoders = {}
    for col in LABEL_ENCODE_COLS:
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"  ✓ Label-encoded '{col}' → {col}_encoded ({len(le.classes_)} classes)")

    # One-hot encode categorical columns
    before_cols = len(df.columns)
    df = pd.get_dummies(df, columns=ONEHOT_COLS, drop_first=False, dtype=int)
    after_cols = len(df.columns)
    print(f"  ✓ One-hot encoded {ONEHOT_COLS} → added {after_cols - before_cols + len(ONEHOT_COLS)} columns")

    return df, label_encoders


# ─── 1.5  Prepare Train/Test Split ──────────────────────────────────────────
def prepare_train_test(df):
    """
    Time-based train-test split.
    Last 20% of dates become the test set.
    """
    print("\n── Train-Test Split (Time-Based) ──")

    sorted_dates = sorted(df["Date"].unique())
    split_idx = int(len(sorted_dates) * (1 - TEST_SIZE_RATIO))
    split_date = sorted_dates[split_idx]

    train_df = df[df["Date"] < split_date].copy()
    test_df = df[df["Date"] >= split_date].copy()

    print(f"  Split date: {pd.Timestamp(split_date).date()}")
    print(f"  Train: {train_df.shape[0]:,} rows ({train_df['Date'].min().date()} → {train_df['Date'].max().date()})")
    print(f"  Test:  {test_df.shape[0]:,} rows ({test_df['Date'].min().date()} → {test_df['Date'].max().date()})")

    # Identify feature columns (everything except identifiers and target)
    drop_cols = [c for c in DROP_COLS_FOR_TRAINING if c in train_df.columns]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET]

    print(f"  Features: {len(feature_cols)} columns")

    return X_train, X_test, y_train, y_test, feature_cols


# ─── Master Pipeline ────────────────────────────────────────────────────────
def load_and_preprocess():
    """
    Run the full preprocessing pipeline.

    Returns:
        df: cleaned + feature-engineered DataFrame (with Date, IDs still intact)
        X_train, X_test, y_train, y_test: train-test split arrays
        feature_cols: list of feature column names
        label_encoders: dict of fitted LabelEncoders
    """
    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)
    df_encoded, label_encoders = encode_features(df.copy())

    # Save processed data for inspection
    df_encoded.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"\n✓ Processed data saved to: {PROCESSED_DATA_PATH}")

    X_train, X_test, y_train, y_test, feature_cols = prepare_train_test(df_encoded)

    print(f"\n{'━' * 60}")
    print(f"  PREPROCESSING COMPLETE")
    print(f"  Final shape: {df_encoded.shape[0]:,} rows × {df_encoded.shape[1]} columns")
    print(f"{'━' * 60}\n")

    return df, df_encoded, X_train, X_test, y_train, y_test, feature_cols, label_encoders


# ─── CLI Entry Point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    df, df_encoded, X_train, X_test, y_train, y_test, feature_cols, _ = load_and_preprocess()
    print("Sample of processed features:")
    print(X_train.head())
    print(f"\nTarget distribution (train): mean={y_train.mean():.1f}, std={y_train.std():.1f}")
    print(f"Target distribution (test):  mean={y_test.mean():.1f}, std={y_test.std():.1f}")
