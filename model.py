import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# Step 1: Load the Data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close']

# Step 2: Preprocess the Data
def preprocess_data(data):
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    # Create the training data
    train_data_len = int(np.ceil(len(scaled_data) * 0.8))
    train_data = scaled_data[:train_data_len]
    test_data = scaled_data[train_data_len - 60:]

    # Create the training dataset
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Create the test dataset
    x_test = []
    y_test = data[train_data_len:]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    return x_train, y_train, x_test, y_test, scaler

# Step 3: Build the LSTM Model
def build_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 4: Train the Model
def train_model(model, x_train, y_train, epochs=10, batch_size=1):
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# Step 5: Make Predictions
def predict(model, x_test, scaler):
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# Step 6: Visualize the Results
def visualize(predictions, y_test):
    data = pd.DataFrame(y_test)
    train = data[:int(len(data) * 0.8)]
    valid = data[int(len(data) * 0.8):]
    valid['Predictions'] = predictions
    
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()

# Main Function
if __name__ == "__main__":
    ticker = 'AAPL'  # Example: Apple Inc.
    start_date = '2010-01-01'
    end_date = '2021-01-01'

    data = load_data(ticker, start_date, end_date)
    x_train, y_train, x_test, y_test, scaler = preprocess_data(data)

    model = build_model()
    train_model(model, x_train, y_train, epochs=10, batch_size=1)

    predictions = predict(model, x_test, scaler)
    visualize(predictions, y_test)