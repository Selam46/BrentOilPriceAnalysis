# Brent Oil Price Analysis Project

## Project Overview
A comprehensive analysis of Brent oil prices, including data processing, statistical modeling, and an interactive dashboard. This project analyzes historical price trends, implements predictive models, and provides a user-friendly interface for exploring oil market dynamics.

## Tasks Overview

### Task 1: Data Processing and Initial Analysis
- **Data Cleaning and Preparation**
  - Loading and processing daily Brent oil prices (1987-2022)
  - Handling missing values and outliers
  - Date format standardization
  - Feature engineering

- **Statistical Analysis**
  - Time series decomposition
  - Trend analysis
  - Seasonality detection
  - Volatility patterns
  - Returns distribution

- **Event Impact Analysis**
  - Major political events correlation
  - Economic crisis impact assessment
  - Market structure breaks identification
  - Volatility clustering analysis

### Task 2: Time Series Modeling
- **Model Implementation**
  - ARIMA (AutoRegressive Integrated Moving Average)
    - Parameter optimization
    - Residual analysis
    - Forecast generation
  
  - SARIMA (Seasonal ARIMA)
    - Seasonal pattern incorporation
    - Parameter tuning
    - Prediction accuracy assessment

- **Model Evaluation**
  - Cross-validation implementation
  - Performance metrics:
    - RMSE (Root Mean Square Error)
    - MAE (Mean Absolute Error)
    - R² Score
  - Residual analysis
  - Model comparison

### Task 3: Interactive Dashboard
- **Frontend Features** (React.js)
  - Interactive price charts
  - Multiple visualization types:
    - Line charts
    - Area charts
    - Volume analysis
    - Technical indicators
  - Event timeline visualization
  - Real-time data filtering
  - Responsive design

- **Backend Implementation** (Flask)
  - RESTful API endpoints
  - Data processing services
  - Model integration
  - Event data management

## Technology Stack
### Frontend
- React.js
- Material-UI
- Recharts
- Emotion styled components

### Backend
- Flask
- Pandas
- NumPy
- Statsmodels

### Data Analysis
- Jupyter Notebooks
- Scikit-learn
- Matplotlib
- Seaborn

## Installation and Setup

### Prerequisites
- Python 3.8+
- Node.js v14+
- pip
- npm

### Data Analysis Environment

pip install -r requirements.txt
- Jupyter Notebooks

### Dashboard Setup

# Backend
cd oil-price-dashboard/backend
pip install -r requirements.txt
python app.py

# Frontend
cd oil-price-dashboard
npm install
npm start

## Usage

### Data Analysis
1. Open Jupyter notebooks in `notebook/` directory
2. Run cells sequentially for analysis and modeling

### Dashboard
1. Start Flask backend server
2. Launch React frontend application
3. Access dashboard at http://localhost:3000

## Features

### Analysis Components
- Time series decomposition
- Event impact analysis
- Volatility clustering
- Return distributions
- Technical indicators

### Models
- ARIMA implementation
- SARIMA modeling
- Cross-validation
- Performance metrics

### Dashboard
- Interactive price charts
- Event timeline
- Technical analysis tools
- Prediction visualization
- Real-time filtering

## Results and Findings
- Model performance comparisons
- Event impact quantification
- Volatility patterns
- Market trend analysis
- Prediction accuracy metrics

## Future Enhancements
- Real-time price updates
- Advanced ML models
- Additional technical indicators
- Enhanced event correlation
- Mobile optimization
- API integrations


## License
MIT License

