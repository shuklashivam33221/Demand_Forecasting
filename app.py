import streamlit as st
import pandas as pd
import joblib
import json
import os
from PIL import Image

# ─── Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Demand Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.5);
    }
    h1, h2, h3 {
        color: #00d4aa !important;
    }
</style>
""", unsafe_allow_html=True)


# Get the directory of the app.py script
APP_DIR = os.path.dirname(os.path.abspath(__file__))


# ─── Helper Functions ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    raw_path = os.path.join(APP_DIR, "demand_forecasting.csv")
    processed_path = os.path.join(APP_DIR, "outputs", "processed_data.csv")
    
    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path, parse_dates=["Date"])
    else:
        # Fallback to raw if processed doesn't exist
        df = pd.read_csv(raw_path, parse_dates=["Date"])
    return df

@st.cache_resource
def load_ml_models():
    models_dir = os.path.join(APP_DIR, "models")
    models = {}
    metadata = {}
    all_results = {}
    
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith(".joblib") and f != "best_model.joblib"]
        for f in model_files:
            name = f.replace(".joblib", "").replace("_", " ").title()
            models[name] = joblib.load(os.path.join(models_dir, f))
        
        # Load best model specifically
        best_model_path = os.path.join(models_dir, "best_model.joblib")
        if os.path.exists(best_model_path):
            models["Best Model"] = joblib.load(best_model_path)
            
        # Load metadata
        metadata_path = os.path.join(models_dir, "model_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                
        # Load all results
        results_path = os.path.join(models_dir, "all_results.json")
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                all_results = json.load(f)
            
    return models, metadata, all_results

def load_image(filename):
    path = os.path.join(APP_DIR, "outputs", filename)
    if os.path.exists(path):
        return Image.open(path)
    return None


# ─── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("📈 Demand Forecasting")
st.sidebar.markdown("Explore data, view EDA, and predict future demand.")

app_mode = st.sidebar.radio(
    "Navigate",
    ["Data Explorer", "EDA Visualizations", "Model Performance", "Time-Series & DL", "Predict Demand"]
)

# ─── Load Data & Models ─────────────────────────────────────────────────────
try:
    df = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.warning("Please run the data preprocessing script first.")
    data_loaded = False

try:
    models, metadata, all_results = load_ml_models()
    models_loaded = len(models) > 0
    if not models_loaded:
        st.sidebar.warning("No models loaded from the models/ directory.")
except Exception as e:
    st.sidebar.error(f"Error loading models: {e}")
    models_loaded = False


# ─── Tab 1: Data Explorer ───────────────────────────────────────────────────
if app_mode == "Data Explorer" and data_loaded:
    st.title("📊 Data Explorer")
    st.markdown("View and filter the demand forecasting dataset.")
    
    # Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Unique Stores", df["Store ID"].nunique() if "Store ID" in df.columns else "N/A")
    with col3:
        st.metric("Unique Products", df["Product ID"].nunique() if "Product ID" in df.columns else "N/A")
    with col4:
        st.metric("Avg Demand", f"{df['Demand'].mean():.1f}" if "Demand" in df.columns else "N/A")
        
    st.markdown("---")
    
    # Filters
    st.subheader("Filter Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if "Date" in df.columns:
            date_range = st.date_input("Date Range", [df["Date"].min(), df["Date"].max()])
        
    with col2:
        if "Category" in df.columns:
            categories = st.multiselect("Select Categories", options=df["Category"].unique())
        elif any(c.startswith("Category_") for c in df.columns):
            st.info("Categories are one-hot encoded.")
            categories = []
        else:
            categories = []
            
    with col3:
        if "Store ID" in df.columns:
            stores = st.multiselect("Select Stores", options=df["Store ID"].unique())
        else:
            stores = []
            
    # Apply filters
    filtered_df = df.copy()
    if "Date" in filtered_df.columns and len(date_range) == 2:
        filtered_df = filtered_df[(filtered_df["Date"].dt.date >= date_range[0]) & (filtered_df["Date"].dt.date <= date_range[1])]
    if categories and "Category" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Category"].isin(categories)]
    if stores and "Store ID" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Store ID"].isin(stores)]
        
    st.dataframe(filtered_df.head(1000), use_container_width=True)
    st.caption(f"Showing {len(filtered_df):,} rows (limited to 1000 for display performance)")


# ─── Tab 2: EDA Visualizations ──────────────────────────────────────────────
elif app_mode == "EDA Visualizations":
    st.title("📈 Exploratory Data Analysis")
    st.markdown("Visual insights extracted from the dataset.")
    
    # Define the list of visualizations
    visualizations = [
        ("01_demand_distribution.png", "Demand Distribution"),
        ("02_demand_over_time.png", "Demand Over Time"),
        ("03_demand_by_category.png", "Demand by Category"),
        ("04_demand_by_region.png", "Demand by Region"),
        ("05_demand_by_season.png", "Demand by Season"),
        ("06_demand_by_weather.png", "Demand by Weather"),
        ("07_promotion_impact.png", "Promotion Impact"),
        ("08_discount_vs_demand.png", "Discount vs Demand"),
        ("09_price_vs_demand.png", "Price vs Demand"),
        ("10_correlation_heatmap.png", "Correlation Heatmap"),
        ("11_store_monthly_trend.png", "Store Monthly Trend"),
        ("12_product_demand_ranking.png", "Top & Bottom Products")
    ]
    
    # Create two columns for displaying images
    col1, col2 = st.columns(2)
    
    for i, (filename, title) in enumerate(visualizations):
        img = load_image(filename)
        if img:
            with col1 if i % 2 == 0 else col2:
                st.subheader(title)
                st.image(img, use_container_width=True)
        else:
            if i == 0:
                st.warning("EDA visualizations not found. Please run `python src/eda.py` first.")
                break


# ─── Tab 3: Model Performance ───────────────────────────────────────────────
elif app_mode == "Model Performance":
    st.title("🤖 Model Performance")
    
    if not models_loaded or not all_results:
        st.warning("Model results not found. Please run `python src/train_model.py` first.")
    else:
        best_model_name = metadata.get("model_name", "Unknown")
        st.markdown(f"### Best Performing Model: **{best_model_name}**")
        
        # Display Metrics Comparison
        st.subheader("Metrics Comparison")
        
        # Format results into a dataframe
        metrics_df = pd.DataFrame(all_results).T
        st.dataframe(metrics_df.style.highlight_min(axis=0, subset=["MAE", "RMSE", "MAPE"])
                                     .highlight_max(axis=0, subset=["R2"]), 
                     use_container_width=True)
        
        # Display Model Visualizations
        st.markdown("---")
        st.subheader("Evaluation Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            img = load_image("13_actual_vs_predicted.png")
            if img:
                st.image(img, caption="Actual vs Predicted Demand", use_container_width=True)
                
            img = load_image("15_feature_importance.png")
            if img:
                st.image(img, caption="Feature Importance", use_container_width=True)
                
        with col2:
            img = load_image("14_residual_distribution.png")
            if img:
                st.image(img, caption="Residual Distribution", use_container_width=True)
                
            img = load_image("16_model_comparison.png")
            if img:
                st.image(img, caption="Model Comparison", use_container_width=True)


# ─── Tab 4: Predict Demand ──────────────────────────────────────────────────
elif app_mode == "Predict Demand":
    st.title("🔮 Predict Future Demand")
    
    if not models_loaded or "Best Model" not in models:
        st.warning("Models not found. Please run `python src/train_model.py` first.")
    else:
        st.markdown("Enter product and market conditions to forecast demand.")
        
        model = models["Best Model"]
        features = metadata.get("features", [])
        
        # Create input form based on feature list
        with st.form("prediction_form"):
            st.subheader("Input Parameters")
            
            col1, col2, col3 = st.columns(3)
            
            input_data = {}
            
            st.info("Adjust the feature values below and click 'Generate Forecast' to predict demand.")
            
            # Dynamically generate inputs for required features
            for i, feature in enumerate(features):
                with [col1, col2, col3][i % 3]:
                    # Determine input type based on feature name
                    if "ratio" in feature or "diff" in feature:
                        input_data[feature] = st.number_input(feature, value=1.0, format="%.2f")
                    elif "is_" in feature or "Promotion" in feature or "Epidemic" in feature:
                        input_data[feature] = st.selectbox(feature, [0, 1])
                    elif "day" in feature or "month" in feature or "week" in feature:
                        input_data[feature] = st.number_input(feature, value=1, min_value=0, step=1)
                    else:
                        input_data[feature] = st.number_input(feature, value=0.0)
            
            submit_button = st.form_submit_button("Generate Forecast")
            
        if submit_button:
            # Create a dataframe from inputs
            input_df = pd.DataFrame([input_data])
            
            # Ensure columns match training features
            for col in features:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            input_df = input_df[features]
            
            try:
                # Make prediction
                prediction = model.predict(input_df)[0]
                
                st.success("Prediction Generated Successfully!")
                
                # Display nicely
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div style="background-color: #1a1c23; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #00d4aa;">
                        <h2 style="margin: 0; color: white;">Forecasted Demand</h2>
                        <h1 style="font-size: 4em; margin: 0; color: #00d4aa;">{max(0, int(prediction))}</h1>
                        <p style="color: #cccccc;">Units</p>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.error("Please ensure all input features match the model's expected format.")


# ─── Tab 5: Time-Series & DL ────────────────────────────────────────────────
elif app_mode == "Time-Series & DL":
    st.title("⏳ Time-Series & Deep Learning Models")
    st.markdown("Forecasting **daily aggregated demand** using time-series and deep learning architectures.")
    
    # Load Time-Series Results
    ts_results_path = os.path.join(APP_DIR, "models", "ts_results.json")
    if os.path.exists(ts_results_path):
        with open(ts_results_path, "r") as f:
            ts_results = json.load(f)
            
        st.subheader("Time-Series Metrics Comparison")
        st.markdown("Models trained on total daily demand sequences. Notice how **XGBoost (with lag features)** outperforms traditional TS (ARIMA/Prophet) and Deep Learning (LSTM) on this specific dataset.")
        
        # Format results into a dataframe
        ts_metrics_df = pd.DataFrame(ts_results).T
        st.dataframe(ts_metrics_df.style.highlight_min(axis=0, subset=["MAE", "RMSE", "MAPE"])
                                     .highlight_max(axis=0, subset=["R2"]), 
                     use_container_width=True)
                     
        st.markdown("---")
        st.subheader("Visualizations")
        
        img_comparison = load_image("17_ts_forecast_comparison.png")
        if img_comparison:
            st.image(img_comparison, caption="Forecast Comparison vs Actual Daily Demand", use_container_width=True)
            
        col1, col2 = st.columns(2)
        with col1:
            img_metrics = load_image("18_ts_model_comparison.png")
            if img_metrics:
                st.image(img_metrics, caption="Model Metrics", use_container_width=True)
        with col2:
            img_res = load_image("19_ts_residuals.png")
            if img_res:
                st.image(img_res, caption="Residuals Distribution", use_container_width=True)
    else:
        st.warning("Time-series results not found. Please run `python src/train_timeseries.py` first.")

