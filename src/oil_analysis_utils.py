import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.preprocessing import StandardScaler

def load_and_process_data(file_path):
    """
    Load and process Brent oil price data with comprehensive cleaning and validation
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing oil price data
        
    Returns:
    --------
    pd.DataFrame
        Processed DataFrame with additional features and cleaned data
        
    Notes:
    ------
    - Handles multiple date formats
    - Removes outliers using IQR method
    - Implements forward filling for missing values
    - Generates technical indicators and statistical features
    """
    # Read the CSV file
    print("Loading data from:", file_path)
    df = pd.read_csv(file_path)
    
    # Document initial data state
    print("\nInitial data shape:", df.shape)
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())
    
    # Date Processing
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
    except ValueError:
        print("\nAttempting alternative date formats...")
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
    
    # Set and sort index
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Remove outliers
    df = remove_outliers(df)
    
    # Generate features
    df = generate_features(df)
    
    # Document final data state
    print("\nFinal data shape:", df.shape)
    print("\nFeatures generated:", list(df.columns))
    
    return df

def handle_missing_values(df):
    """
    Handle missing values using sophisticated methods
    
    Strategy:
    1. Forward fill for up to 5 days
    2. Linear interpolation for longer gaps
    3. Document all modifications
    """
    missing_before = df.isnull().sum()
    
    # Forward fill short gaps
    df['Price'] = df['Price'].fillna(method='ffill', limit=5)
    
    # Interpolate longer gaps
    df['Price'] = df['Price'].interpolate(method='linear')
    
    # Document changes
    missing_after = df.isnull().sum()
    print("\nMissing values handled:")
    print(f"Before: {missing_before}")
    print(f"After: {missing_after}")
    
    return df

def remove_outliers(df, threshold=3):
    """
    Remove outliers using IQR method with documentation
    """
    initial_rows = len(df)
    
    # Calculate IQR
    Q1 = df['Price'].quantile(0.25)
    Q3 = df['Price'].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    # Remove outliers
    df = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)]
    
    # Document changes
    removed_rows = initial_rows - len(df)
    print(f"\nOutliers removed: {removed_rows} rows")
    print(f"Outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    return df

def generate_features(df):
    """
    Generate comprehensive feature set for analysis
    
    Features:
    1. Basic statistical features
    2. Technical indicators
    3. Temporal features
    4. Volatility measures
    """
    # Price changes and returns
    df['Returns'] = df['Price'].pct_change()
    df['Log_Returns'] = np.log(df['Price']/df['Price'].shift(1))
    
    # Moving averages
    for window in [5, 10, 20, 50, 200]:
        df[f'MA_{window}'] = df['Price'].rolling(window=window).mean()
        df[f'Returns_MA_{window}'] = df['Returns'].rolling(window=window).mean()
    
    # Volatility measures
    df['Daily_Volatility'] = df['Returns'].rolling(window=20).std()
    df['Annual_Volatility'] = df['Daily_Volatility'] * np.sqrt(252)
    
    # Technical indicators
    df['RSI'] = calculate_rsi(df['Price'])
    df['MACD'], df['MACD_Signal'] = calculate_macd(df['Price'])
    
    # Momentum indicators
    df['Momentum'] = df['Price'] / df['Price'].shift(10) - 1
    
    # Temporal features
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day_of_Week'] = df.index.dayofweek
    
    return df

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def perform_stationarity_tests(df):
    """
    Perform comprehensive stationarity analysis
    
    Tests:
    1. Augmented Dickey-Fuller
    2. KPSS test
    3. Visual analysis components
    """
    results = {}
    
    # ADF Test
    adf_result = adfuller(df['Price'])
    results['ADF'] = {
        'test_statistic': adf_result[0],
        'p_value': adf_result[1],
        'critical_values': adf_result[4]
    }
    
    # KPSS Test
    kpss_result = kpss(df['Price'])
    results['KPSS'] = {
        'test_statistic': kpss_result[0],
        'p_value': kpss_result[1]
    }
    
    # Decomposition
    decomposition = seasonal_decompose(df['Price'], period=252)
    results['decomposition'] = decomposition
    
    return results

def analyze_event_impact(df, events_dict):
    """
    Analyze price behavior around significant events
    
    Parameters:
    -----------
    df : DataFrame
        Processed price data
    events_dict : dict
        Dictionary of events with dates and descriptions
    
    Returns:
    --------
    DataFrame
        Impact analysis results for each event
    """
    results = []
    window = 30  # Analysis window (days) around event
    
    for event, date in events_dict.items():
        try:
            # Get data around event
            start_date = pd.to_datetime(date) - pd.Timedelta(days=window)
            end_date = pd.to_datetime(date) + pd.Timedelta(days=window)
            event_data = df[start_date:end_date]
            
            # Calculate metrics
            pre_event_price = event_data['Price'][:window].mean()
            post_event_price = event_data['Price'][-window:].mean()
            price_change = ((post_event_price - pre_event_price) / pre_event_price) * 100
            
            volatility_change = (
                event_data['Daily_Volatility'][-window:].mean() / 
                event_data['Daily_Volatility'][:window].mean() - 1
            ) * 100
            
            results.append({
                'Event': event,
                'Date': date,
                'Price_Impact_%': price_change,
                'Volatility_Change_%': volatility_change,
                'Recovery_Days': calculate_recovery_time(event_data, pre_event_price)
            })
            
        except Exception as e:
            print(f"Error analyzing event {event}: {str(e)}")
    
    return pd.DataFrame(results)

def calculate_recovery_time(event_data, pre_event_price):
    """Calculate days until price recovers to pre-event level"""
    event_middle = len(event_data) // 2
    post_event_data = event_data.iloc[event_middle:]
    recovery_point = post_event_data[post_event_data['Price'] >= pre_event_price].index
    
    if len(recovery_point) > 0:
        return (recovery_point[0] - event_data.index[event_middle]).days
    return None

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