import streamlit as st
import yfinance as yf
import pandas as pd

# Function to download the S&P 500 data
def load_data():
    ticker = "^GSPC"  # S&P 500 symbol
    data = yf.download(ticker, start="2020-01-01", end="2025-01-01")
    return data

# Function to make a simple prediction based on the last two closing prices
def simple_prediction(data):
    last_close = data['Close'].iloc[-1]  # Last closing price
    previous_close = data['Close'].iloc[-2]  # Previous closing price
    
    if last_close > previous_close:
        return "UP"
    else:
        return "DOWN"

# Streamlit layout for the app
st.title("S&P 500 Prediction App")
st.write("This simple app predicts if the S&P 500 will go UP or DOWN based on the last two closing prices.")

# Load data
data = load_data()

# Display the latest data (optional)
st.write("### Latest S&P 500 Data", data.tail())

# Make a simple prediction
prediction = simple_prediction(data)

# Display prediction
st.write(f"The prediction for tomorrow's S&P 500 movement is: **{prediction}**")
