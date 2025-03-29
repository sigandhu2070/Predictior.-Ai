# Predictior.-Ai
Stock Market Prediction AI (Google Cloud Deployment)

Folder Structure

/stock-ai-gcloud
│── app.py              # Streamlit App for Stock Prediction
│── requirements.txt    # Required Python Libraries
│── Dockerfile          # Docker Configuration for Google Cloud Run
│── README.md           # Setup and Deployment Guide

1. app.py (Stock Prediction AI)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import streamlit as st

def get_stock_data(ticker, start, end):
    stock_data = yf.download(ticker, start=start, end=end)
    return stock_data[['Close']]

def prepare_data(stock_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data)
    X_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        X_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train, scaler

def build_lstm_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_rf_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    return model

def build_xgb_model(X_train, y_train):
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    return model

def predict_stock_price(model, scaler, stock_data, model_type='lstm'):
    inputs = scaler.transform(stock_data[-60:].values)
    inputs = np.reshape(inputs, (1, inputs.shape[0], 1))
    predicted_price = model.predict(inputs) if model_type == 'lstm' else model.predict(inputs.reshape(1, -1))
    return scaler.inverse_transform(predicted_price.reshape(-1, 1))

def main():
    st.title("Stock Market Prediction AI")
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, GOOGL, MSFT):", "AAPL")
    if st.button("Predict Stock Price"):
        stock_data = get_stock_data(ticker, "2020-01-01", "2024-01-01")
        X_train, y_train, scaler = prepare_data(stock_data)
        lstm_model = build_lstm_model()
        lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        rf_model = build_rf_model(X_train, y_train)
        xgb_model = build_xgb_model(X_train, y_train)
        lstm_pred = predict_stock_price(lstm_model, scaler, stock_data, 'lstm')
        rf_pred = predict_stock_price(rf_model, scaler, stock_data, 'rf')
        xgb_pred = predict_stock_price(xgb_model, scaler, stock_data, 'xgb')
        st.write(f"Predicted stock price for {ticker} (LSTM): {lstm_pred[0][0]:.2f}")
        st.write(f"Predicted stock price for {ticker} (Random Forest): {rf_pred[0][0]:.2f}")
        st.write(f"Predicted stock price for {ticker} (XGBoost): {xgb_pred[0][0]:.2f}")
if __name__ == "__main__":
    main()

2. requirements.txt

numpy
pandas
matplotlib
yfinance
scikit-learn
xgboost
tensorflow
streamlit
gunicorn

3. Dockerfile

FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.enableCORS=false"]

4. README.md (Google Cloud Deployment Guide)

# Stock Market Prediction AI

## Steps to Deploy on Google Cloud Run

1. **Set up Google Cloud Project**
   ```sh
   gcloud config set project YOUR_PROJECT_ID

2. Build Docker Image

gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/stock-ai


3. Deploy to Cloud Run

gcloud run deploy stock-ai --image gcr.io/YOUR_PROJECT_ID/stock-ai --platform managed --region us-central1 --allow-unauthenticated


4. Access the App (Google Cloud will provide a URL)



