import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

def load_and_process_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Print sample of dates to debug
    print("Sample of dates before conversion:")
    print(df['Date'].head())
    print("\nUnique date formats in the dataset:")
    print(df['Date'].unique()[:5])
    
    try:
        # First attempt with the expected format
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
    except ValueError as e:
        print(f"First attempt failed with error: {e}")
        try:
            # Second attempt with dayfirst=True
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        except ValueError as e:
            print(f"Second attempt failed with error: {e}")
            # Third attempt with format inference
            df['Date'] = pd.to_datetime(df['Date'], format='mixed')
    
    # Set Date as index
    df.set_index('Date', inplace=True)
    
    # Sort index
    df.sort_index(inplace=True)
    
    # Calculate daily returns
    df['Returns'] = df['Price'].pct_change()
    
    # Calculate volatility (20-day rolling standard deviation)
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    return df

def analyze_time_series(df):
    # Decompose the time series
    decomposition = seasonal_decompose(df['Price'], period=252)  # 252 trading days in a year
    
    # Perform Augmented Dickey-Fuller test
    adf_result = adfuller(df['Price'])
    
    # Calculate summary statistics
    stats = df['Price'].describe()
    
    return decomposition, adf_result, stats 