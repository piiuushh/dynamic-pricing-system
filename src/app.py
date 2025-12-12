import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from datetime import datetime, timedelta
import time
import os

# Import our modules
from generator import generate_synthetic_data
from features import create_features
from model import train_model, load_model, save_model
from optimization import optimize_price

st.set_page_config(page_title="AI Dynamic Pricing", layout="wide", page_icon="💎")

# --- Professional UI CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    :root {
        --white: #ffffff;
        --off-white: #f8f9fa;
        --light-grey: #e9ecef;
        --grey: #6c757d;
        --dark-grey: #495057;
        --black: #212529;
        --blue: #0d6efd;
        --cyan: #0dcaf0;
        --blue-light: #d0e7ff;
        --shadow: rgba(0, 0, 0, 0.1);
    }
    
    * {
        transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
    }
    
    .main {
        background: var(--white);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: var(--black);
    }
    
    .stSidebar {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-right: 3px solid var(--blue);
    }
    
    .stSidebar [data-testid="stMarkdownContainer"] p,
    .stSidebar label {
        color: var(--black) !important;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        color: var(--blue) !important;
    }
    
    /* Metric Cards */
    [data-testid="stMetric"] {
        background: var(--white);
        padding: 1.75rem 1.5rem;
        border-radius: 16px;
        border: 2px solid var(--light-grey);
        box-shadow: 0 4px 12px var(--shadow), 0 1px 3px rgba(0, 0, 0, 0.05);
        position: relative;
        overflow: hidden;
    }
    
    [data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--blue), var(--cyan));
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    [data-testid="stMetric"]:hover::before {
        opacity: 1;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: var(--blue);
        box-shadow: 0 8px 24px rgba(13, 110, 253, 0.15), 0 4px 8px var(--shadow);
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--grey) !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.25rem !important;
        font-weight: 800 !important;
        font-family: 'Inter', sans-serif;
        color: var(--black);
    }
    
    [data-testid="stMetric"]:nth-child(1) [data-testid="stMetricValue"] {
        background: linear-gradient(135deg, var(--blue) 0%, var(--cyan) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    [data-testid="stMetric"]:nth-child(2) [data-testid="stMetricValue"] {
        background: linear-gradient(135deg, #0d6efd 0%, #6ea8fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    [data-testid="stMetric"]:nth-child(3) [data-testid="stMetricValue"] {
        background: linear-gradient(135deg, var(--cyan) 0%, #6edff6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    [data-testid="stMetric"]:nth-child(4) [data-testid="stMetricValue"] {
        background: linear-gradient(135deg, #495057 0%, var(--grey) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Typography */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.75rem;
        color: var(--black);
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
        border-bottom: 4px solid var(--blue);
        padding-bottom: 0.75rem;
    }
    
    h2 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 1.5rem;
        color: var(--blue);
        margin-top: 2.5rem;
        margin-bottom: 1rem;
        letter-spacing: -0.01em;
    }
    
    h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        color: var(--dark-grey);
        letter-spacing: 0.01em;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--off-white);
        padding: 8px;
        border-radius: 12px;
        border: 2px solid var(--light-grey);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background: var(--white);
        border-radius: 10px;
        color: var(--grey);
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0 24px;
        border: 2px solid var(--light-grey);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--blue-light);
        color: var(--blue);
        border-color: var(--blue);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--blue), var(--cyan));
        color: white !important;
        border-color: var(--blue);
        box-shadow: 0 4px 12px rgba(13, 110, 253, 0.25);
    }
    
    /* Input Fields */
    .stTextInput input, .stSelectbox select, .stTextArea textarea {
        background: var(--white) !important;
        border: 2px solid var(--light-grey) !important;
        border-radius: 10px;
        color: var(--black) !important;
        padding: 0.75rem 1rem;
        font-size: 0.95rem;
    }
    
    .stTextInput input:focus, .stSelectbox select:focus, .stTextArea textarea:focus {
        border-color: var(--blue) !important;
        box-shadow: 0 0 0 3px rgba(13, 110, 253, 0.1) !important;
        outline: none;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, var(--blue) 0%, var(--cyan) 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.75rem;
        font-weight: 700;
        font-size: 0.95rem;
        box-shadow: 0 4px 12px rgba(13, 110, 253, 0.3);
        letter-spacing: 0.02em;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(13, 110, 253, 0.4);
        background: linear-gradient(135deg, #0b5ed7 0%, #0bb5d4 100%);
    }
    
    /* Alerts */
    .stAlert {
        background: var(--off-white);
        border: 2px solid var(--light-grey);
        border-left: 4px solid var(--blue);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        color: var(--black);
    }
    
    .stSuccess {
        border-left-color: #198754;
        background: #d1e7dd;
    }
    
    .stWarning {
        border-left-color: #ffc107;
        background: #fff3cd;
    }
    
    .stError {
        border-left-color: #dc3545;
        background: #f8d7da;
    }
    
    .stInfo {
        border-left-color: var(--cyan);
        background: #cff4fc;
    }
    
    /* Radio & Checkboxes */
    .stRadio > label, .stCheckbox > label {
        color: var(--black) !important;
        font-weight: 600;
    }
    
    .stRadio [role="radiogroup"] label,
    .stCheckbox [role="checkbox"] + div {
        color: var(--dark-grey) !important;
    }
    
    /* Sliders */
    .stSlider {
        padding: 1rem 0;
    }
    
    .stSlider > label {
        color: var(--black) !important;
        font-weight: 600;
    }
    
    /* Selectbox */
    .stSelectbox label {
        color: var(--black);
        font-weight: 600;
    }
    
    /* Markdown & Text */
    .stMarkdown {
        color: var(--dark-grey);
    }
    
    p, li, span {
        color: var(--dark-grey);
    }
    
    /* Dividers */
    hr {
        border-color: var(--light-grey);
        margin: 2rem 0;
    }
    
    /* Code blocks */
    code {
        background: var(--off-white);
        color: var(--blue);
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        border: 1px solid var(--light-grey);
    }
    
    pre {
        background: var(--off-white);
        border: 1px solid var(--light-grey);
        border-radius: 8px;
    }
    
    /* Dataframes & Tables */
    .stDataFrame {
        background: var(--white);
        border: 2px solid var(--light-grey);
        border-radius: 12px;
    }
    
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background: var(--white);
    }
    
    /* Expander */
    .stExpander {
        background: var(--white);
        border: 1px solid var(--light-grey);
        border-radius: 10px;
    }
    
    .stExpander [data-testid="stExpanderToggleIcon"] {
        color: var(--blue);
    }
    
    /* File Uploader */
    .stFileUploader {
        background: var(--off-white);
        border: 2px dashed var(--light-grey);
        border-radius: 10px;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--blue), var(--cyan));
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: var(--blue);
    }
    
    /* Container spacing */
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(use_csv, data_source):
    if use_csv:
        if data_source == "Indian Market Data (CSV)" and os.path.exists("data/india_ride_data.csv"):
             return pd.read_csv("data/india_ride_data.csv")
        elif os.path.exists("data/sample_ride_data.csv"):
            return pd.read_csv("data/sample_ride_data.csv")
        else:
            return None
    return None

@st.cache_resource
def get_model(use_csv=False, data_source="Synthetic Simulation"):
    model_path = "model_csv.joblib" if use_csv else "model_synthetic.joblib"
    
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        with st.spinner(f"Training model ({'CSV' if use_csv else 'Synthetic'})..."):
            try:
                if use_csv:
                    df = load_data(use_csv, data_source)
                    if df is None:
                        st.error("Data file not found.")
                        return None

                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    # Ensure columns exist
                    if 'traffic' not in df.columns: df['traffic'] = 'High' # Default for India
                else:
                    df = generate_synthetic_data(n_samples=2000)
                
                df = create_features(df)
                model, metrics = train_model(df)
                save_model(model, model_path)
                return model
            except Exception as e:
                st.error(f"Error during model training: {e}")
                return None

def main():
    st.title("AI-Driven Dynamic Pricing System")
    st.markdown("### Intelligent Fare Optimization Engine")
    
    # Sidebar Controls
    with st.sidebar:
        st.header("Configuration")
        
        # Data Source Selector
        # Data Source Selector
        data_source = st.radio("Training Data Source", ["Synthetic Simulation", "Real Data (CSV)", "Indian Market Data (CSV)"])
        use_csv = (data_source in ["Real Data (CSV)", "Indian Market Data (CSV)"])
        
        if use_csv:
            st.success("Using `data/sample_ride_data.csv`")
        
        st.divider()
        
        st.header("Simulation Control")
        
        st.subheader("Contextual Factors")
        # Time of Day Slider
        time_of_day = st.slider("Time of Day (24h)", 0, 23, 18)
        weather = st.selectbox("Weather Condition", ["Clear", "Rainy", "Foggy"])
        event_status = st.selectbox("Event Status", ["None", "Concert", "Sports", "Festival"])
        
        st.subheader("Market Dynamics")
        n_requests = st.slider("Active Requests", 10, 300, 120)
        
        st.info("Adjust sliders to simulate different market conditions.")

    # Generate "Real-time" Context
    current_date = datetime.now().replace(hour=time_of_day, minute=0, second=0)
    
    # Data Generation
    selected_city = "All" # Default value
    search_area = "" # Default value
    if data_source == "Indian Market Data (CSV)" and os.path.exists("data/india_ride_data.csv"):
        # Load a sample from the Indian dataset for visualization
        df_full = load_data(True, data_source)
        # Filter for a specific city if needed, or just take a random sample
        # Let's add a city selector for the map if Indian data is active
        with st.sidebar:
            st.subheader("Location Filter")
            selected_city = st.selectbox("Select City", ["All", "Delhi", "Mumbai", "Bangalore"])
            search_area = st.text_input("Search Area", placeholder="e.g., Connaught Place")
        
        # Apply filters to full dataset
        if search_area:
            # Case-insensitive search
            df_sim = df_full[df_full['location_name'].str.contains(search_area, case=False, na=False)]
            if df_sim.empty:
                st.sidebar.warning(f"No data found for '{search_area}'. Showing random sample.")
                if selected_city != "All":
                    df_sim = df_full[df_full['city'] == selected_city].sample(n=min(n_requests, len(df_full[df_full['city'] == selected_city])))
                else:
                    df_sim = df_full.sample(n=min(n_requests, len(df_full)))
            else:
                st.sidebar.success(f"Found {len(df_sim)} rides in '{search_area}'")
                # If too many results, sample down to n_requests to keep UI snappy, unless n_requests is small
                if len(df_sim) > n_requests:
                    df_sim = df_sim.sample(n=n_requests)
        elif selected_city != "All":
            df_sim = df_full[df_full['city'] == selected_city].sample(n=min(n_requests, len(df_full[df_full['city'] == selected_city])))
        else:
            df_sim = df_full.sample(n=min(n_requests, len(df_full)))
            
        # Ensure timestamp is datetime
        df_sim['timestamp'] = pd.to_datetime(df_sim['timestamp'])
            
        # Ensure we have lat/long
        if 'latitude' not in df_sim.columns:
            st.error("Indian data missing coordinates.")
            df_sim = generate_synthetic_data(n_samples=n_requests, start_date=current_date)
    else:
        # Default synthetic generation
        df_sim = generate_synthetic_data(n_samples=n_requests, start_date=current_date)
        df_sim['weather'] = weather
        df_sim['event'] = event_status
    
    # Feature Engineering
    df_features = create_features(df_sim)
    
    # Model Inference
    model = get_model(use_csv=use_csv, data_source=data_source)
    
    if model is None:
        st.error("Failed to load or train the model. Please check the logs.")
        return
    
    # Representative Data (City Center)
    city_center_data = df_features[df_features['location_name'] == 'City Center']
    row = city_center_data.iloc[0] if not city_center_data.empty else df_features.iloc[0]
        
    input_data = pd.DataFrame([row])
    input_data = input_data[['demand_ratio', 'is_rainy', 'is_peak_hour', 'is_city_center', 'is_weekend']]
    
    predicted_multiplier = model.predict(input_data)[0]
    
    # Optimization
    current_utilization = min(1.0, row['demand_ratio'])
    optimized_multiplier = optimize_price(predicted_multiplier, current_utilization)
    
    
    # Top Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Demand Ratio", f"{row['demand_ratio']:.2f}", delta="High" if row['demand_ratio'] > 1.2 else "Normal")
    with col2:
        st.metric("Driver Utilization", f"{current_utilization*100:.1f}%", delta_color="inverse")
    with col3:
        st.metric("AI Predicted Surge", f"{predicted_multiplier:.2f}x")
    with col4:
        delta = optimized_multiplier - predicted_multiplier
        st.metric("Optimized Surge", f"{optimized_multiplier:.2f}x", delta=f"{delta:.2f}", delta_color="normal")


    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Live Map", "Analytics & Insights", "Model Performance"])
    
    with tab1:
        c1, c2 = st.columns([3, 1])
        with c1:
            # Map Style Selector
            map_style = st.selectbox("Map Style", ["Light", "Dark", "Satellite", "Road"], index=0)
            
            map_styles = {
                "Light": "light",
                "Dark": "dark",
                "Satellite": "satellite",
                "Road": "road"
            }
            

            
            # PyDeck Map
            layer = pdk.Layer(
                "ScatterplotLayer",
                df_sim,
                get_position=["longitude", "latitude"],
                get_color=[200, 30, 0, 160],
                get_radius=100,
                pickable=True,
            )
            
            # Determine View State based on data
            if not df_sim.empty:
                mid_lat = df_sim['latitude'].mean()
                mid_long = df_sim['longitude'].mean()
                # Zoom in if searching or a specific city is selected
                if search_area:
                    zoom_level = 14
                elif selected_city != "All":
                    zoom_level = 12
                else:
                    zoom_level = 10
            else:
                mid_lat = 28.6139
                mid_long = 77.2090
                zoom_level = 10

            view_state = pdk.ViewState(
                latitude=mid_lat,
                longitude=mid_long,
                zoom=zoom_level,
                pitch=50,
            )
            
            st.pydeck_chart(pdk.Deck(
                map_style=map_styles[map_style],
                layers=[layer],
                initial_view_state=view_state,
                tooltip={
                    "html": "<b>{location_name}</b><br/>"
                            "Requests: {requests}<br/>"
                            "Drivers: {drivers}<br/>"
                            "Weather: {weather}<br/>"
                            "Base Fare: {base_fare}<br/>",
                    "style": {
                        "backgroundColor": "steelblue",
                        "color": "white"
                    }
                }
            ))
        
        with c2:
            st.markdown("#### Pricing Decision")
            
            # Gauge Chart for Surge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = optimized_multiplier,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Surge Multiplier"},
                gauge = {
                    'axis': {'range': [1, 5]},
                    'bar': {'color': "#FF4B4B"},
                    'steps': [
                        {'range': [1, 1.5], 'color': "lightgreen"},
                        {'range': [1.5, 2.5], 'color': "yellow"},
                        {'range': [2.5, 5], 'color': "red"}],
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.info(f"Base Fare: ₹{row['base_fare']}")
            st.success(f"Final Fare: ₹{row['base_fare'] * optimized_multiplier:.2f}")

    with tab2:
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("Demand vs Supply Trend (Simulated)")
            # Simulate a 24h trend
            hours = list(range(24))
            demand_trend = [50 + 20*np.sin(h/24 * 2*np.pi) + np.random.normal(0, 5) for h in hours]
            supply_trend = [40 + 15*np.sin(h/24 * 2*np.pi) + np.random.normal(0, 5) for h in hours]
            
            df_trend = pd.DataFrame({"Hour": hours, "Demand": demand_trend, "Supply": supply_trend})
            fig_trend = px.line(df_trend, x="Hour", y=["Demand", "Supply"], title="24H Market Trend")
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with col_b:
            st.subheader("Price Sensitivity Analysis")
            # Show how price changes with demand ratio
            ratios = np.linspace(0.5, 3.0, 50)
            prices = [1.0 + max(0, (r - 1.5) * 0.5) for r in ratios] # Simple logic replication
            df_sens = pd.DataFrame({"Demand Ratio": ratios, "Price Multiplier": prices})
            fig_sens = px.area(df_sens, x="Demand Ratio", y="Price Multiplier", title="Surge Logic Curve")
            st.plotly_chart(fig_sens, use_container_width=True)

    with tab3:
        st.subheader("Model Diagnostics")
        
        # We need to re-train or load metrics to show them here
        # For performance, we'll just re-calculate on a small batch or store them
        # Let's generate a fresh batch for validation
        df_val = generate_synthetic_data(n_samples=500)
        df_val = create_features(df_val)
        
        # Get feature importance from the model (if available)
        if hasattr(model, "feature_importances_"):
            feat_imp = pd.DataFrame({
                "Feature": ['demand_ratio', 'is_rainy', 'is_peak_hour', 'is_city_center', 'is_weekend'],
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)
            
            fig_imp = px.bar(feat_imp, x="Importance", y="Feature", orientation='h', title="Feature Importance")
            st.plotly_chart(fig_imp)
        
        st.write("Note: Model is retrained periodically on new data.")

if __name__ == "__main__":
    main()
