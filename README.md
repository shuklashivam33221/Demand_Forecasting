# 📈 Demand Forecasting — ML Project

An end-to-end machine learning project that predicts retail product demand using historical sales data, pricing signals, weather conditions, promotions, and seasonality. The project includes a complete data pipeline, exploratory analysis, model training, and an interactive Streamlit dashboard deployed with a live URL.

🔗 **Live Dashboard:** [demandforecasting-pp8akythieaa3mtd2wtgdt.streamlit.app](https://demandforecasting-pp8akythieaa3mtd2wtgdt.streamlit.app)

---

## 🎯 Problem Statement

Retailers need accurate demand forecasts to optimize inventory, reduce waste, and maximize revenue. This project builds a supervised regression pipeline that learns from 76,000+ historical records across 5 stores and 20 products to predict future unit demand.

---

## 📊 Dataset Overview

| Property | Value |
|---|---|
| **Rows** | 76,000 |
| **Date Range** | Jan 2022 – Jan 2024 |
| **Stores** | 5 |
| **Products** | 20 |
| **Categories** | Electronics, Clothing, Furniture, Groceries, Toys |
| **Target Variable** | `Demand` (units) |

**Key Features:** Price, Discount, Competitor Pricing, Inventory Level, Units Sold, Units Ordered, Weather Condition, Promotion, Seasonality, Epidemic flag, Region.

---

## 🏗️ Project Structure

```
Demand Forecasting model/
│
├── demand_forecasting.csv        # Raw dataset (76K rows)
├── requirements.txt              # Python dependencies
├── app.py                        # Streamlit dashboard (deployed)
├── README.md
├── .gitignore
│
├── src/
│   ├── utils.py                  # Shared constants, metrics, helpers
│   ├── data_preprocessing.py     # Data cleaning & feature engineering
│   ├── eda.py                    # 12 EDA visualizations
│   └── train_model.py            # Model training & evaluation
│
├── outputs/                      # Generated plots & processed data
│   ├── 01_demand_distribution.png
│   ├── 02_demand_over_time.png
│   ├── ...                       # 12 EDA + 4 model evaluation plots
│   └── processed_data.csv
│
└── models/                       # Trained models & metadata
    ├── best_model.joblib          # XGBoost (best performer)
    ├── xgboost.joblib
    ├── linear_regression.joblib
    ├── model_metadata.json
    └── all_results.json
```

---

## ⚙️ Pipeline Overview

### Phase 1 — Data Preprocessing (`src/data_preprocessing.py`)
- Fixes data quality issues (e.g., `Fzurniture` → `Furniture` typo)
- Handles missing values and duplicates
- **12 engineered features:**
  - *Time-based:* day_of_week, day_of_month, month, week_of_year, is_weekend, is_month_start, is_month_end
  - *Price-based:* price_discount_ratio, price_competitor_diff, price_competitor_ratio
  - *Inventory-based:* inventory_sold_ratio, sold_ordered_ratio
- Label-encodes Store/Product IDs; one-hot encodes Category, Region, Weather, Seasonality
- **Time-based train-test split** (80/20 by date, split at Sep 2023)

### Phase 2 — Exploratory Data Analysis (`src/eda.py`)
Generates 12 publication-quality, dark-themed visualizations:

| # | Chart | Insight |
|---|---|---|
| 1 | Demand Distribution | Near-normal distribution, mean ≈ 104 units |
| 2 | Demand Over Time | Monthly aggregated trend line |
| 3 | Demand by Category | Box plots across 5 product categories |
| 4 | Demand by Region | Regional demand comparison |
| 5 | Demand by Season | Seasonal demand patterns |
| 6 | Demand by Weather | Weather condition impact |
| 7 | Promotion Impact | Promo vs non-promo violin plot |
| 8 | Discount vs Demand | Scatter with regression line |
| 9 | Price vs Demand | Price-demand relationship |
| 10 | Correlation Heatmap | Feature correlation matrix |
| 11 | Store Monthly Trend | Per-store time series |
| 12 | Product Ranking | Top & bottom 5 products by demand |

### Phase 3 — Model Training (`src/train_model.py`)
Trains and evaluates 4 regression models with sensible defaults (no expensive hyperparameter tuning):

| Model | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| Linear Regression | 15.56 | 20.69 | 0.7798 | 19.39% |
| Random Forest | 12.10 | 15.81 | 0.8713 | 13.29% |
| Gradient Boosting | 10.96 | 14.33 | 0.8944 | 12.27% |
| **XGBoost** | **9.97** | **13.08** | **0.9120** | **11.28%** |

> **Best Model: XGBoost** — R² = 0.912, off by only ~10 units on average.

Generates 4 evaluation visualizations: Actual vs Predicted scatter, Residual distribution, Feature Importance bar chart, and Model Comparison dashboard.

### Phase 4 — Interactive Dashboard (`app.py`)
A Streamlit web app with 4 sections:
- **Data Explorer** — Filter by date, category, and store with live summary metrics
- **EDA Visualizations** — Browse all 12 analysis charts in a two-column gallery
- **Model Performance** — Metrics comparison table with highlighted best values + evaluation plots
- **Predict Demand** — Input feature values and get a real-time XGBoost demand forecast

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+

### Installation

```bash
git clone https://github.com/shuklashivam33221/Demand_Forecasting.git
cd Demand_Forecasting
pip install -r requirements.txt
```

### Run the Pipeline

```bash
python src/data_preprocessing.py   # Clean data & engineer features
python src/eda.py                  # Generate EDA visualizations
python src/train_model.py          # Train models & save best
```

### Launch the Dashboard Locally

```bash
streamlit run app.py
```

---

## 🌐 Deployment

The dashboard is deployed on **Streamlit Community Cloud** and is publicly accessible:

🔗 **[https://demandforecasting-pp8akythieaa3mtd2wtgdt.streamlit.app](https://demandforecasting-pp8akythieaa3mtd2wtgdt.streamlit.app)**

To redeploy after changes:
1. Commit and push to the `main` branch on GitHub.
2. Streamlit Cloud automatically detects changes and redeploys.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Dashboard** | Streamlit |
| **Deployment** | Streamlit Community Cloud |
| **Version Control** | Git, GitHub |

---

## 📁 Key Files

| File | Description |
|---|---|
| `src/utils.py` | Project paths, evaluation metrics (MAE, RMSE, R², MAPE), metadata I/O |
| `src/data_preprocessing.py` | Full cleaning + feature engineering + encoding + time-based split |
| `src/eda.py` | 12 dark-themed EDA visualizations saved as PNGs |
| `src/train_model.py` | Trains 4 models, selects best by MAE, saves with joblib |
| `app.py` | Streamlit dashboard with 4 interactive sections |
| `requirements.txt` | All Python dependencies |

---

## 📄 License

This project is open source and available for educational and portfolio purposes.
