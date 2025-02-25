import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv('../../data/BrentOilPrices.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df['Volatility'] = df['Price'].pct_change().rolling(window=20).std()
    return df

def process_events():
    return {
        'political': [
            {'date': '2022-02-24', 'event': 'Russia-Ukraine War', 'impact': 'High'},
            {'date': '2019-09-14', 'event': 'Saudi Oil Facility Attack', 'impact': 'Medium'},
            # Add more events...
        ],
        'economic': [
            {'date': '2020-03-15', 'event': 'COVID-19 Pandemic', 'impact': 'High'},
            {'date': '2008-09-15', 'event': 'Global Financial Crisis', 'impact': 'High'},
            # Add more events...
        ]
    }

def calculate_metrics(df):
    return {
        'average_price': float(df['Price'].mean()),
        'volatility': float(df['Price'].std()),
        'max_drawdown': float(calculate_max_drawdown(df['Price'])),
        'current_price': float(df['Price'].iloc[-1]),
        'price_change': float(df['Price'].pct_change().iloc[-1] * 100)
    }

def calculate_max_drawdown(prices):
    peak = prices.expanding(min_periods=1).max()
    drawdown = (prices - peak) / peak
    return drawdown.min() * 100