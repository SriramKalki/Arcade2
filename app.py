from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

app = Flask(__name__)

# Function to load and preprocess data
def load_and_preprocess_data(ticker):
    data = yf.download(ticker, start="2010-01-01", end="2023-01-01")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return data, scaled_data, scaler

# Function to create LSTM model
def create_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.json['ticker']
    data, scaled_data, scaler = load_and_preprocess_data(ticker)
    
    # Prepare the test data
    test_data = scaled_data[-120:]  # last 120 days for prediction
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    # Create and train the model
    model = create_model()
    model.fit(x_test, scaled_data[-60:], epochs=10, batch_size=1, verbose=0)
    
    # Make prediction
    prediction = model.predict(x_test[-1].reshape(1, 60, 1))
    prediction = scaler.inverse_transform(prediction)
    
    result = {
        'ticker': ticker,
        'prediction': float(prediction[0][0])
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
    # begin 