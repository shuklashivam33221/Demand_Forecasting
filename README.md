# Demand Forecasting ML Project

This project predicts future product demand based on historical sales data, pricing, weather, seasonality, and promotional activities. It includes a complete machine learning pipeline and an interactive Streamlit dashboard.

## Features
- **Data Preprocessing:** Cleans data, fixes typos, handles missing values, and generates 12+ derived features (time-based, price ratios, inventory stats).
- **Exploratory Data Analysis (EDA):** Generates 12 professional visualizations to uncover insights and trends in the dataset.
- **Model Training:** Trains and evaluates 4 ML models (Linear Regression, Random Forest, Gradient Boosting, XGBoost) using a time-based split, and saves the best-performing model based on Mean Absolute Error (MAE).
- **Interactive Dashboard:** A stunning, dark-themed Streamlit app with 4 tabs: Data Explorer, EDA Visualizations, Model Performance, and Predict Demand.

## Project Structure
```
demand_forecasting/
├── demand_forecasting.csv          # Raw dataset (76,000 rows)
├── requirements.txt                # Python dependencies
├── src/
│   ├── utils.py                    # Shared metrics and helpers
│   ├── data_preprocessing.py       # Data cleaning & feature engineering
│   ├── eda.py                      # Visualization generation
│   └── train_model.py              # Model training & evaluation
├── outputs/                        # Generated EDA plots and processed data
├── models/                         # Trained models and metadata
└── app.py                          # Streamlit dashboard
```

## Setup Instructions

1. **Install Dependencies:**
   Ensure you have Python 3.9+ installed. Run the following command:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Pipeline:**
   Execute the scripts in order to generate data, plots, and models:
   ```bash
   python src/data_preprocessing.py
   python src/eda.py
   python src/train_model.py
   ```

3. **Start the Dashboard:**
   Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Deployment (Streamlit Community Cloud)

To get a live, shareable URL for this dashboard:
1. Create a GitHub repository and push this code.
2. Sign up at [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
3. Click "New app", select your repository, branch, and set `app.py` as the Main file path.
4. Click "Deploy"!
