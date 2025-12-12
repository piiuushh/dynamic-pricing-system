# Technical Reference: AI-Driven Dynamic Pricing System

## 1. System Architecture

The system follows a modular architecture designed for flexibility and scalability.

### Core Components

- **Data Generator (`src/generator.py`)**:
  - Simulates a ride-sharing environment.
  - Generates synthetic locations (lat/lon) clustered around city centers.
  - Simulates demand (requests) and supply (drivers) based on time of day and weather conditions.
  - Calculates `base_fare` and `distance` for each ride.

- **Feature Engineering (`src/features.py`)**:
  - Transforms raw simulation data into model-ready features.
  - Key features:
    - `demand_ratio`: Requests / Drivers.
    - `time_of_day`: Hour of the day (0-23).
    - `is_peak_hour`: Boolean flag for rush hours (8-10 AM, 5-8 PM).
    - `is_weekend`: Boolean flag.
    - `weather_condition`: Encoded categorical variable (Clear, Rain, Storm).

- **Machine Learning Model (`src/model.py`)**:
  - **Algorithm**: XGBoost Regressor.
  - **Target**: `price_multiplier` (Optimal surge factor).
  - **Training**:
    - Uses historical (or synthetic) data to learn the relationship between market conditions and optimal pricing.
    - Metrics: RMSE, R2 Score.
  - **Persistence**: Models are saved as `.joblib` files for inference.

- **Optimization Engine (`src/optimization.py`)**:
  - **Library**: PuLP (Linear Programming).
  - **Objective**: Maximize Revenue = `Price * Predicted_Conversion_Rate`.
  - **Constraints**:
    - `Min Surge`: 1.0x (No price gouging below base).
    - `Max Surge`: 5.0x (Regulatory cap).
    - `Utilization Floor`: Ensure prices don't drop so low that drivers leave.
    - `Demand Smoothing`: Prevent extreme price jumps between consecutive intervals.

- **Frontend (`src/app.py`)**:
  - Built with **Streamlit**.
  - Handles user interaction, visualization, and real-time simulation control.
  - Integrates **PyDeck** for geospatial visualization and **Plotly** for analytics.

## 2. Data Flow

1.  **Input**: User adjusts simulation parameters (Time, Weather, City) via the Sidebar.
2.  **Simulation**: `generator.py` creates a snapshot of the market (Requests vs Drivers).
3.  **Processing**: `features.py` calculates the `demand_ratio` and other derived metrics.
4.  **Prediction**: `model.py` uses the pre-trained XGBoost model to predict a baseline surge multiplier.
5.  **Optimization**: `optimization.py` takes the predicted surge and current utilization to output the final `optimized_multiplier`.
6.  **Output**: The dashboard updates with the new price, map visualization, and KPIs.

## 3. Key Algorithms

### 3.1 Demand-Supply Logic
The simulation uses a probabilistic model to determine demand:
```python
base_demand = 100
if is_peak_hour:
    base_demand *= 1.5
if weather == "Rain":
    base_demand *= 1.2
```

### 3.2 Optimization Function
We solve for `x` (final price multiplier):
$$ \text{Maximize } R(x) = x \cdot D(x) $$
Where $D(x)$ is the demand function, modeled as a decaying function of price:
$$ D(x) = D_{base} \cdot e^{-\alpha(x-1)} $$
Subject to:
$$ 1.0 \le x \le 5.0 $$

## 4. Docker Configuration

The application is containerized using a multi-stage build process.

- **Base Image**: `python:3.10-slim` for a lightweight footprint.
- **Dependencies**: Installed via `requirements.txt`. System dependencies like `build-essential` are included for compiling Python packages.
- **Port**: Exposes port `8501` for Streamlit.
- **Healthcheck**: Includes a curl-based healthcheck to ensure the service is responsive.

## 5. Future Extensibility

- **Database Integration**: Replace in-memory CSV handling with PostgreSQL/Redis.
- **API Layer**: Expose the pricing engine via FastAPI for external consumption.
- **A/B Testing**: Implement a framework to test different pricing strategies live.
