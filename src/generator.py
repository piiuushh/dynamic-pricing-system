import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(n_samples=1000, start_date=None):
    """
    Generates synthetic data for the dynamic pricing system using vectorized operations.
    
    Args:
        n_samples (int): Number of samples to generate.
        start_date (datetime): Starting timestamp.
        
    Returns:
        pd.DataFrame: DataFrame containing synthetic ride data.
    """
    if start_date is None:
        start_date = datetime.now()
        
    # Locations: keys and data
    loc_names = ["City Center", "Suburbs", "Airport"]
    loc_coords = {
        "City Center": (28.6139, 77.2090),
        "Suburbs": (28.5355, 77.3910),
        "Airport": (28.5562, 77.1000)
    }
    
    # Generate Timestamps
    # Vectorized time addition
    minutes_offsets = np.arange(n_samples) * 15
    timestamps = []
    # For very large offsets, direct numpy addition to datetime can be tricky with types, 
    # but for 100k * 15 mins = 1.5 million mins = ~3 years it fits in timedelta64[m]
    
    # Faster way: base + timedelta array
    time_deltas = np.array([timedelta(minutes=int(x)) for x in minutes_offsets]) # List comp still fastest for mixed types often, but let's try pure numpy if possible.
    # Actually for 100k, list comp is fine.
    timestamps = np.array([start_date + timedelta(minutes=int(x)) for x in minutes_offsets])
    
    hours = np.array([t.hour for t in timestamps])
    
    # Generate Locations
    chosen_loc_indices = np.random.randint(0, len(loc_names), n_samples)
    chosen_locs = np.array(loc_names)[chosen_loc_indices]
    
    # Get base lat/longs
    base_lats = np.array([loc_coords[loc][0] for loc in chosen_locs])
    base_longs = np.array([loc_coords[loc][1] for loc in chosen_locs])
    
    # Add noise
    lats = base_lats + np.random.normal(0, 0.01, n_samples)
    longs = base_longs + np.random.normal(0, 0.01, n_samples)
    
    # Base Demand
    base_demand = np.full(n_samples, 50.0)
    
    # Peak hours: 8-10, 17-20
    is_peak = ((hours >= 8) & (hours <= 10)) | ((hours >= 17) & (hours <= 20))
    base_demand[is_peak] *= 2.0
    
    # City Center boost
    base_demand[chosen_locs == "City Center"] *= 1.5
    
    # Weather
    weather_conditions = ["Clear", "Rainy", "Foggy"]
    weather_probs = [0.7, 0.2, 0.1]
    chosen_weather = np.random.choice(weather_conditions, n_samples, p=weather_probs)
    
    base_demand[chosen_weather == "Rainy"] *= 1.3
    
    # Event
    event_types = ["None", "Concert", "Sports", "Festival"]
    event_probs = [0.9, 0.03, 0.03, 0.04]
    chosen_events = np.random.choice(event_types, n_samples, p=event_probs)
    
    base_demand[chosen_events != "None"] *= 1.5
    
    # Final Requests
    requests = np.random.poisson(base_demand)
    
    # Supply (Drivers)
    base_drivers = np.full(n_samples, 40.0)
    base_drivers[is_peak] *= 1.2
    base_drivers[chosen_weather == "Rainy"] *= 0.8
    
    drivers = np.random.poisson(base_drivers)
    drivers = np.maximum(1, drivers) # Avoid zero
    
    # Traffic
    traffic_conditions = ["Low", "Medium", "High"]
    chosen_traffic = np.random.choice(traffic_conditions, n_samples)
    
    # Base Fare
    base_fare = 100 + np.random.randint(-10, 21, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "location_name": chosen_locs,
        "latitude": lats,
        "longitude": longs,
        "requests": requests,
        "drivers": drivers,
        "weather": chosen_weather,
        "traffic": chosen_traffic,
        "event": chosen_events,
        "base_fare": base_fare
    })
    
    return df

if __name__ == "__main__":
    import time
    start = time.time()
    df = generate_synthetic_data(n_samples=100000)
    end = time.time()
    print(f"Generated {len(df)} rows in {end-start:.4f} seconds.")
    print(df.head())
