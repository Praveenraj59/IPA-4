import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os
import pandas as pd


# Create directory for saved models if it doesn't exist
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

scaler = MinMaxScaler(feature_range=(0, 1))

# Fetch historical stock data from Yahoo Finance
def get_stock_data(ticker, start_date='2018-01-01', end_date=None):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# Prepare data for LSTM
def prepare_data(data, time_step=60):
    data = data['Close'].values.reshape(-1, 1)
    data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])

    return np.array(X), np.array(y)

# Build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the LSTM model
def train_lstm(ticker):
    stock_data = get_stock_data(ticker)
    if stock_data is None or stock_data.empty:
        print(f"No data available for {ticker}")
        return None, None, None

    time_step = 60
    X, y = prepare_data(stock_data, time_step)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = build_lstm_model((X.shape[1], 1))
    model.fit(X, y, epochs=20, batch_size=32, verbose=1)

    model.save(f'saved_models/{ticker}_lstm.h5')
    joblib.dump(scaler, f'saved_models/{ticker}_scaler.pkl')

    return stock_data

# Predict stock prices
def predict_stock_price(ticker):
    model_path = f'saved_models/{ticker}_lstm.h5'
    scaler_path = f'saved_models/{ticker}_scaler.pkl'

    stock_data = get_stock_data(ticker)
    if stock_data is None or stock_data.empty:
        print(f"No data available for {ticker}")
        return None, "No stock data available.", None

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Training model for {ticker} as no saved model was found.")
        stock_data = train_lstm(ticker)
        if stock_data is None or stock_data.empty:
            return None, "Failed to train model or get data.", None

    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return None, "Error loading saved model or scaler.", None

    data = stock_data['Close'].values.reshape(-1, 1)
    data = scaler.transform(data)

    time_step = 60
    if len(data) <= time_step:
        return None, "Not enough data for prediction.", None

    X_test = []
    for i in range(time_step, len(data)):
        X_test.append(data[i-time_step:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    actual_prices = stock_data['Close'][-len(predictions):]
    predicted_prices = predictions.flatten()

    predicted_price = float(predicted_prices[-1])  # Ensures it’s a single float


    return predicted_price, actual_prices, predicted_prices
