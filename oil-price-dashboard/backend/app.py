from flask import Flask, jsonify
from flask_cors import CORS
from utils.data_processor import load_data, process_events, calculate_metrics

app = Flask(__name__)
CORS(app)

# Load and process data once at startup
df = load_data()
events_data = process_events()
metrics = calculate_metrics(df)

@app.route('/api/prices', methods=['GET'])
def get_prices():
    return jsonify({
        'dates': df.index.strftime('%Y-%m-%d').tolist(),
        'prices': df['Price'].tolist(),
        'volatility': df['Volatility'].tolist()
    })

@app.route('/api/events', methods=['GET'])
def get_events():
    return jsonify(events_data)

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    return jsonify(metrics)

if __name__ == '__main__':
    app.run(debug=True)