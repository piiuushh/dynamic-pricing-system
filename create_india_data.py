import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_india_csv(filename="data/india_ride_data.csv", n_samples=5000):
    """
    Creates a realistic Indian market dataset for the dynamic pricing system.
    Cities: Delhi, Mumbai, Bangalore.
    """
    np.random.seed(100) # Different seed
    
    start_date = datetime.now() - timedelta(days=60)
    data = []
    
    cities = ["Delhi", "Mumbai", "Bangalore"]
    # Location types within cities
    # Location types within cities with approximate Lat/Long
    locations = {
        "Delhi": {
            "Connaught Place": (28.6304, 77.2177),
            "Gurgaon Cyber Hub": (28.4950, 77.0895),
            "IGI Airport": (28.5562, 77.1000),
            "Saket": (28.5244, 77.2188),
            "Noida Sector 18": (28.5708, 77.3270)
        },
        "Mumbai": {
            "Bandra Kurla Complex": (19.0674, 72.8681),
            "Nariman Point": (18.9256, 72.8242),
            "CSMT": (18.9415, 72.8355),
            "Andheri": (19.1136, 72.8697),
            "Juhu Beach": (19.0988, 72.8264)
        },
        "Bangalore": {
            "Koramangala": (12.9352, 77.6245),
            "Indiranagar": (12.9716, 77.6412),
            "Whitefield": (12.9698, 77.7500),
            "Electronic City": (12.8399, 77.6770),
            "MG Road": (12.9766, 77.5993)
        }
    }
    
    weather_conditions = ["Clear", "Monsoon Rain", "Smog/Haze", "Humid"]
    events = ["None", "IPL Match", "Diwali Rush", "Office Rush", "Wedding Season"]
    
    for i in range(n_samples):
        timestamp = start_date + timedelta(minutes=np.random.randint(0, 60*24*60))
        hour = timestamp.hour
        day = timestamp.weekday()
        month = timestamp.month
        
        city = np.random.choice(cities)
        loc_name = np.random.choice(list(locations[city].keys()))
        base_lat, base_long = locations[city][loc_name]
        
        # Add small noise to coordinates for scattering
        lat = base_lat + np.random.normal(0, 0.005)
        long = base_long + np.random.normal(0, 0.005)
        
        # Base demand logic
        base_demand = 60
        
        # City specific logic
        if city == "Bangalore" and (loc_name in ["Whitefield", "Electronic City"]):
            # Tech hubs have massive peak hour spikes
            if 8 <= hour <= 11 or 17 <= hour <= 20: base_demand += 80
        
        if city == "Mumbai" and (loc_name == "Bandra Kurla Complex"):
            if 17 <= hour <= 19: base_demand += 70
            
        if city == "Delhi" and (loc_name == "Gurgaon Cyber Hub"):
             if 17 <= hour <= 20: base_demand += 75

        # Time impact
        if 18 <= hour <= 21: base_demand *= 1.6 # Evening peak in India is intense
        if day >= 5: base_demand *= 1.3 # Weekend
        
        # Weather - Monsoon impact
        weather = np.random.choice(weather_conditions, p=[0.6, 0.2, 0.1, 0.1])
        if weather == "Monsoon Rain": 
            base_demand *= 1.8 # Rain in Indian cities causes chaos
        
        # Events
        event = np.random.choice(events, p=[0.85, 0.03, 0.02, 0.08, 0.02])
        if event == "IPL Match" and 19 <= hour <= 23: base_demand *= 2.0
        if event == "Diwali Rush": base_demand *= 2.5
        if event == "Office Rush" and (8 <= hour <= 11 or 17 <= hour <= 21): base_demand *= 1.5
        
        requests = int(np.random.poisson(base_demand))
        
        # Supply logic - Drivers
        # Supply is often lower than demand in peak hours in India
        supply_factor = np.random.uniform(0.5, 1.1)
        if weather == "Monsoon Rain": supply_factor *= 0.6 # Drivers go offline
        if event == "Diwali Rush": supply_factor *= 0.7
        
        drivers = int(requests * supply_factor)
        drivers = max(1, drivers)
        
        # Pricing Logic (Historical)
        base_fare = 50 # Lower base fare in India
        if city == "Mumbai": base_fare = 60
        
        # Distance factor (simulated)
        distance_km = np.random.uniform(2, 25)
        fare_per_km = 12
        if city == "Bangalore": fare_per_km = 14
        
        total_base_fare = base_fare + (distance_km * fare_per_km)
        
        demand_ratio = requests / drivers
        surge = 1.0
        
        # Aggressive surge in India
        if demand_ratio > 1.5: surge += (demand_ratio - 1.5) * 1.0
        if weather == "Monsoon Rain": surge += 0.5
        if event == "IPL Match": surge += 0.8
        
        # Cap surge
        surge = min(surge, 4.0)
        
        final_price = total_base_fare * surge
        
        data.append({
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "city": city,
            "location_name": loc_name,
            "latitude": lat,
            "longitude": long,
            "requests": requests,
            "drivers": drivers,
            "weather": weather,
            "event": event,
            "base_fare": round(total_base_fare, 2),
            "actual_price_paid": round(final_price, 2)
        })
        
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Created {filename} with {n_samples} rows.")

if __name__ == "__main__":
    create_india_csv()
