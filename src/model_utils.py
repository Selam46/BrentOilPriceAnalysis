import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesModels:
    """
    Comprehensive time series modeling class implementing multiple models
    with detailed documentation and performance analysis
    """
    
    def __init__(self, df, target_col='Price', test_size=0.2):
        """
        Initialize with data and configuration
        
        Parameters:
        -----------
        df : DataFrame
            Input data with datetime index
        target_col : str
            Target column for prediction
        test_size : float
            Proportion of data for testing
        """
        self.df = df
        self.target_col = target_col
        self.test_size = test_size
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        
        # Prepare data
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare data for modeling with comprehensive feature engineering"""
        # Split data
        train_size = int(len(self.df) * (1 - self.test_size))
        self.train = self.df[:train_size]
        self.test = self.df[train_size:]
        
        # Scale data for LSTM
        scaler = StandardScaler()
        self.train_scaled = scaler.fit_transform(self.train[[self.target_col]])
        self.test_scaled = scaler.transform(self.test[[self.target_col]])
        self.scaler = scaler
        
        print(f"Training data shape: {self.train.shape}")
        print(f"Testing data shape: {self.test.shape}")
        
    def build_arima(self, order=(1,1,1)):
        """Build and train ARIMA model"""
        print("\nTraining ARIMA model...")
        model = ARIMA(self.train[self.target_col], order=order)
        self.models['ARIMA'] = model.fit()
        self.predictions['ARIMA'] = self.models['ARIMA'].forecast(len(self.test))
        self.evaluate_model('ARIMA')
        
    def build_sarima(self, order=(1,1,1), seasonal_order=(1,1,1,12)):
        """Build and train SARIMA model"""
        print("\nTraining SARIMA model...")
        model = SARIMAX(self.train[self.target_col], 
                       order=order, 
                       seasonal_order=seasonal_order)
        self.models['SARIMA'] = model.fit()
        self.predictions['SARIMA'] = self.models['SARIMA'].forecast(len(self.test))
        self.evaluate_model('SARIMA')
        
    def build_lstm(self, lookback=60):
        """
        Build and train LSTM model
        
        Parameters:
        -----------
        lookback : int
            Number of previous time steps to use as input features
        """
        print("\nTraining LSTM model...")
        
        # Prepare sequences
        X, y = self.prepare_sequences(self.train_scaled, lookback)
        X_test, y_test = self.prepare_sequences(self.test_scaled, lookback)
        
        # Build model
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(lookback, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Train model
        history = model.fit(X, y, 
                          epochs=100, 
                          batch_size=32,
                          validation_split=0.1,
                          verbose=0)
        
        # Make predictions
        self.models['LSTM'] = model
        lstm_pred = model.predict(X_test)
        self.predictions['LSTM'] = self.scaler.inverse_transform(lstm_pred)
        self.evaluate_model('LSTM')
        
    def prepare_sequences(self, data, lookback):
        """Prepare sequences for LSTM"""
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:(i + lookback)])
            y.append(data[i + lookback])
        return np.array(X), np.array(y)
        
    def evaluate_model(self, model_name):
        """
        Evaluate model performance with multiple metrics
        """
        actual = self.test[self.target_col]
        pred = self.predictions[model_name]
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(actual, pred)),
            'MAE': mean_absolute_error(actual, pred),
            'R2': r2_score(actual, pred)
        }
        
        self.metrics[model_name] = metrics
        print(f"\n{model_name} Model Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
    def cross_validate(self, n_splits=5):
        """
        Perform time series cross-validation
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = {model: {'RMSE': [], 'MAE': [], 'R2': []} 
                    for model in self.models.keys()}
        
        for train_idx, test_idx in tscv.split(self.df):
            cv_train = self.df.iloc[train_idx]
            cv_test = self.df.iloc[test_idx]
            
            for model_name in self.models.keys():
                if model_name == 'LSTM':
                    # Handle LSTM separately
                    continue
                
                # Train and predict
                if model_name == 'ARIMA':
                    model = ARIMA(cv_train[self.target_col], order=(1,1,1))
                else:  # SARIMA
                    model = SARIMAX(cv_train[self.target_col], 
                                  order=(1,1,1), 
                                  seasonal_order=(1,1,1,12))
                
                fitted = model.fit()
                predictions = fitted.forecast(len(cv_test))
                
                # Calculate metrics
                cv_scores[model_name]['RMSE'].append(
                    np.sqrt(mean_squared_error(cv_test[self.target_col], predictions)))
                cv_scores[model_name]['MAE'].append(
                    mean_absolute_error(cv_test[self.target_col], predictions))
                cv_scores[model_name]['R2'].append(
                    r2_score(cv_test[self.target_col], predictions))
        
        # Print cross-validation results
        print("\nCross-validation Results:")
        for model_name, scores in cv_scores.items():
            print(f"\n{model_name}:")
            for metric, values in scores.items():
                mean_score = np.mean(values)
                std_score = np.std(values)
                print(f"{metric}: {mean_score:.4f} (+/- {std_score:.4f})")