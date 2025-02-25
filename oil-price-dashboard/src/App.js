import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import './App.css';

function App() {
  const [priceData, setPriceData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [timeRange, setTimeRange] = useState('1Y');

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/prices');
        const data = await response.json();
        
        // Transform data for Recharts
        const formattedData = data.dates.map((date, index) => ({
          date: date,
          price: data.prices[index],
          volatility: data.volatility[index]
        }));
        
        setPriceData(formattedData);
        setLoading(false);
      } catch (err) {
        setError('Failed to fetch data');
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) return <div className="loading">Loading...</div>;
  if (error) return <div className="error">{error}</div>;

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>Brent Oil Price Analysis Dashboard</h1>
        <div className="time-range-selector">
          <button 
            className={timeRange === '1M' ? 'active' : ''} 
            onClick={() => setTimeRange('1M')}
          >
            1M
          </button>
          <button 
            className={timeRange === '6M' ? 'active' : ''} 
            onClick={() => setTimeRange('6M')}
          >
            6M
          </button>
          <button 
            className={timeRange === '1Y' ? 'active' : ''} 
            onClick={() => setTimeRange('1Y')}
          >
            1Y
          </button>
          <button 
            className={timeRange === 'ALL' ? 'active' : ''} 
            onClick={() => setTimeRange('ALL')}
          >
            ALL
          </button>
        </div>
      </header>

      <div className="chart-container">
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={priceData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="date" 
              tick={{ fontSize: 12 }}
              tickFormatter={(value) => new Date(value).toLocaleDateString()}
            />
            <YAxis 
              yAxisId="price"
              tick={{ fontSize: 12 }}
              domain={['auto', 'auto']}
              label={{ value: 'Price (USD)', angle: -90, position: 'insideLeft' }}
            />
            <YAxis 
              yAxisId="volatility"
              orientation="right"
              tick={{ fontSize: 12 }}
              domain={[0, 'auto']}
              label={{ value: 'Volatility', angle: 90, position: 'insideRight' }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Line 
              yAxisId="price"
              type="monotone" 
              dataKey="price" 
              stroke="#1a73e8" 
              dot={false}
              name="Price"
            />
            <Line 
              yAxisId="volatility"
              type="monotone" 
              dataKey="volatility" 
              stroke="#34a853" 
              dot={false}
              name="Volatility"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="metrics-panel">
        <div className="metric-card">
          <h3>Current Price</h3>
          <p>${priceData[priceData.length - 1].price.toFixed(2)}</p>
        </div>
        <div className="metric-card">
          <h3>Volatility</h3>
          <p>{(priceData[priceData.length - 1].volatility * 100).toFixed(2)}%</p>
        </div>
      </div>
    </div>
  );
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload) return null;

  return (
    <div className="custom-tooltip">
      <p className="date">{new Date(label).toLocaleDateString()}</p>
      <p className="price">Price: ${payload[0].value.toFixed(2)}</p>
      <p className="volatility">Volatility: {(payload[1].value * 100).toFixed(2)}%</p>
    </div>
  );
}

export default App;