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


def analyze_event_periods(df, event_periods):
    """
    Analyze oil prices during specific event periods
    
    Parameters:
    df: DataFrame with oil prices
    event_periods: dict with event details {name: (start_date, end_date)}
    """
    results = {}
    
    for event_name, (start_date, end_date) in event_periods.items():
        # Extract period data
        period_data = df.loc[start_date:end_date]
        
        # Calculate metrics
        metrics = {
            'avg_price': period_data['Price'].mean(),
            'price_volatility': period_data['Returns'].std() * np.sqrt(252),  # Annualized
            'price_change': (period_data['Price'][-1] - period_data['Price'][0]) / period_data['Price'][0] * 100,
            'max_drawdown': calculate_max_drawdown(period_data['Price']),
            'avg_daily_volume': period_data['Returns'].abs().mean() * 100
        }
        
        results[event_name] = metrics
    
    return pd.DataFrame(results).T

def calculate_max_drawdown(prices):
    """Calculate the maximum drawdown percentage"""
    peak = prices.expanding(min_periods=1).max()
    drawdown = (prices - peak) / peak
    return drawdown.min() * 100

def identify_structural_breaks(df, window=63):  # ~3 months
    """
    Identify potential structural breaks in the price series
    """
    # Calculate rolling mean and standard deviation
    roll_mean = df['Price'].rolling(window=window).mean()
    roll_std = df['Price'].rolling(window=window).std()
    
    # Calculate z-scores
    z_scores = (df['Price'] - roll_mean) / roll_std
    
    # Identify significant breaks (z-score > 2 or < -2)
    breaks = z_scores[abs(z_scores) > 2]
    
    return breaks