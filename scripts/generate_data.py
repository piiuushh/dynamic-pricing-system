import argparse
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.generator import generate_synthetic_data

def generate_sample_data(n_samples, output_file):
    print(f"Generating {n_samples} rows of SAMPLE data...")
    # Use the vectorized generator from src which defaults to sample-like data
    df = generate_synthetic_data(n_samples=n_samples)
    
    # Calculate price for sample data
    # Logic: Base + Surge
    df['demand_ratio'] = df['requests'] / df['drivers']
    
    # Vectorized price calculation
    surge = np.ones(n_samples)
    surge[df['demand_ratio'] > 1.2] += (df.loc[df['demand_ratio'] > 1.2, 'demand_ratio'] - 1.2) * 0.8
    surge[df['weather'] == 'Rainy'] += 0.2
    
    df['actual_price_paid'] = round(df['base_fare'] * surge, 2)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

def generate_india_data(n_samples, output_file):
    print(f"Generating {n_samples} rows of INDIA data...")
    # Custom generator logic for India (reuse minimal parts or rewrite vectorized)
    # To keep performance high, we should implement a vectorized update or specialized generator.
    # For now, let's use the main generator structure but patch the values.
    
    df = generate_synthetic_data(n_samples=n_samples)
    
    # Overwrite Locations
    cities = ["Delhi", "Mumbai", "Bangalore"]
    city_locs = {
        "Delhi": ["Connaught Place", "Gurgaon Cyber Hub", "IGI Airport"],
        "Mumbai": ["Bandra Kurla Complex", "Nariman Point", "Andheri"],
        "Bangalore": ["Koramangala", "Indiranagar", "Electronic City"]
    }
    
    chosen_cities = np.random.choice(cities, n_samples)
    
    # Map cities to specific locations
    # Vectorized mapping is tricky without a join. 
    # Fast approach: loop over cities
    generated_locs = np.empty(n_samples, dtype=object)
    
    for city in cities:
        mask = chosen_cities == city
        count = np.sum(mask)
        if count > 0:
            locs = np.random.choice(city_locs[city], count)
            generated_locs[mask] = locs
            
    df['city'] = chosen_cities
    df['location_name'] = generated_locs
    
    # India specific adjustments
    df['base_fare'] = 50 + np.random.randint(0, 30, n_samples)
    
    # Adjust Demand based on events (IPL etc)
    # Simple multiplier for demo
    df['demand_ratio'] = df['requests'] / df['drivers']
    
    # India Pricing
    surge = np.ones(n_samples)
    surge[df['demand_ratio'] > 1.5] += (df.loc[df['demand_ratio'] > 1.5, 'demand_ratio'] - 1.5) * 1.0
    surge[df['weather'] == 'Rainy'] += 0.5
    surge = np.minimum(surge, 4.0)
    
    df['actual_price_paid'] = round(df['base_fare'] * surge, 2)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic ride data.")
    parser.add_argument("--type", choices=["sample", "india"], default="sample", help="Type of data to generate")
    parser.add_argument("--count", type=int, default=5000, help="Number of rows to generate")
    parser.add_argument("--output", type=str, default=None, help="Output filename")
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f"data/{args.type}_ride_data.csv"
        
    if args.type == "sample":
        generate_sample_data(args.count, args.output)
    elif args.type == "india":
        generate_india_data(args.count, args.output)
