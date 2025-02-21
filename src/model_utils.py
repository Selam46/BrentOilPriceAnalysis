import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

def prepare_data_for_modeling(df, test_size=0.2):
    """Prepare data for modeling"""
    # Create features
    df['MA5'] = df['Price'].rolling(window=5).mean()
    df['MA20'] = df['Price'].rolling(window=20).mean()
    df['MA50'] = df['Price'].rolling(window=50).mean()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    df['Price_Lag1'] = df['Price'].shift(1)
    df['Price_Lag5'] = df['Price'].shift(5)
    
    # Drop NaN values
    df = df.dropna()
    
    # Split the data
    train_size = int(len(df) * (1 - test_size))
    train = df[:train_size]
    test = df[train_size:]
    
    return train, test

def build_arima_model(train, test):
    """Build and evaluate ARIMA model"""
    # Fit ARIMA model
    model = ARIMA(train['Price'], order=(1,1,1))
    results = model.fit()
    
    # Make predictions
    predictions = results.forecast(len(test))
    
    return predictions, results

def build_sarima_model(train, test):
    """Build and evaluate SARIMA model"""
    # Fit SARIMA model
    model = SARIMAX(train['Price'], 
                    order=(1, 1, 1), 
                    seasonal_order=(1, 1, 1, 12))
    results = model.fit()
    
    # Make predictions
    predictions = results.forecast(len(test))
    
    return predictions, results

def evaluate_models(test, predictions_dict):
    """Evaluate multiple models"""
    results = {}
    for name, pred in predictions_dict.items():
        mse = mean_squared_error(test['Price'], pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test['Price'], pred)
        r2 = r2_score(test['Price'], pred)
        
        results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    
    return pd.DataFrame(results).T

def plot_residuals_analysis(model_results, title):
    """Plot residuals analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Residuals over time
    residuals = model_results.resid
    ax1.plot(residuals)
    ax1.set_title('Residuals Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Residual')
    
    # Residuals histogram
    residuals.hist(ax=ax2, bins=30)
    ax2.set_title('Residuals Distribution')
    
    # Q-Q plot
    QQ = ProbPlot(residuals)
    QQ.qqplot(line='45', ax=ax3)
    ax3.set_title('Q-Q Plot')
    
    # Autocorrelation plot
    plot_acf(residuals, ax=ax4)
    ax4.set_title('Autocorrelation Plot')
    
    plt.suptitle(f'{title} - Residuals Analysis')
    plt.tight_layout()
    return fig

def time_series_cv(df, n_splits=5):
    """Perform time series cross-validation"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = {
        'ARIMA': [],
        'SARIMA': []
    }
    
    for train_idx, test_idx in tscv.split(df):
        train_cv = df.iloc[train_idx]
        test_cv = df.iloc[test_idx]
        
        # ARIMA
        arima_pred_cv, _ = build_arima_model(train_cv, test_cv)
        arima_rmse = np.sqrt(mean_squared_error(test_cv['Price'], arima_pred_cv))
        cv_scores['ARIMA'].append(arima_rmse)
        
        # SARIMA
        sarima_pred_cv, _ = build_sarima_model(train_cv, test_cv)
        sarima_rmse = np.sqrt(mean_squared_error(test_cv['Price'], sarima_pred_cv))
        cv_scores['SARIMA'].append(sarima_rmse)
    
    return cv_scores