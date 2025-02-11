import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Custom CSS for the colorful design
st.markdown("""
    <style>
        body {
            background-color: #f0f8ff;
            color: #333;
            font-family: Arial, sans-serif;
        }
        .title {
            color: #2e8b57;
            font-size: 40px;
            font-weight: bold;
            text-align: center;
        }
        .header {
            background-color: #4682b4;
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .prediction-box {
            background-color: #ffcccb;
            padding: 15px;
            border-radius: 10px;
            color: #8b0000;
            font-size: 20px;
        }
        .options-box {
            background-color: #e6f7ff;
            padding: 15px;
            border-radius: 10px;
            color: #333;
            font-size: 20px;
        }
        .footer {
            text-align: center;
            font-size: 18px;
            color: #4682b4;
            margin-top: 30px;
        }
        .button {
            background-color: #4682b4;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 18px;
            margin: 10px;
            border: none;
            cursor: pointer;
        }
        .button:hover {
            background-color: #5f9ea0;
        }
    </style>
""", unsafe_allow_html=True)

# Function to download SPY data
def load_data():
    ticker = "SPY"  # SPY ETF symbol
    today = datetime.today().strftime('%Y-%m-%d')  # Get today's date dynamically
    data = yf.download(ticker, start="2018-01-01", end=today)  # Get the past 5 years of data
    return data

# Function to make a simple prediction based on the last two closing prices
def simple_prediction(data):
    last_close = data['Close'].iloc[-1]
    previous_close = data['Close'].iloc[-2]
    
    # Convert to float (to make sure we're working with scalar numbers)
    last_close = float(last_close)
    previous_close = float(previous_close)
    
    # Predict an upward or downward movement for the next few days (till the end of the week)
    if last_close > previous_close:
        return "UP"
    else:
        return "DOWN"

# Function to get SPY options contracts and generate strike price and expiration date
def get_spy_options():
    ticker = "SPY"
    spy = yf.Ticker(ticker)
    
    # Get the expiration dates for the options
    expirations = spy.options
    next_friday = datetime.today() + timedelta((4 - datetime.today().weekday()) % 7)  # Get next Friday
    expiration_str = next_friday.strftime('%Y-%m-%d')
    
    # If next Friday is an available expiration date, use it, otherwise fallback to a random available expiration
    if expiration_str in expirations:
        expiration_date = expiration_str
    else:
        expiration_date = expirations[0]
    
    # Get call and put options data for the expiration date
    options_data = spy.option_chain(expiration_date)
    calls = options_data.calls
    puts = options_data.puts
    
    # Select the closest strike price around the current price
    current_price = spy.history(period="1d")['Close'].iloc[-1]
    closest_call_strike = calls.loc[(calls['strike'] - current_price).abs().idxmin()]
    closest_put_strike = puts.loc[(puts['strike'] - current_price).abs().idxmin()]
    
    return {
        'call_option': closest_call_strike['strike'],
        'put_option': closest_put_strike['strike'],
        'expiration_date': expiration_date
    }

# Streamlit layout for the app
st.markdown('<p class="title">SPY ETF Prediction App</p>', unsafe_allow_html=True)
st.write("This simple app predicts if the SPY ETF will go UP or DOWN based on the last two closing prices.")

# Add a refresh button to update the data
if st.button('Refresh Data', key='refresh', help="Click to refresh the data"):
    data = load_data()
    st.write("### Latest Data Refreshed")
else:
    data = load_data()

# Display prediction and options button
st.markdown('<p class="header">### SPY ETF Prediction for the Week</p>', unsafe_allow_html=True)

# Make a simple prediction for the trend
prediction = simple_prediction(data)

# Display prediction in a styled box
st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
st.write(f"The prediction for SPY ETF movement till the end of this week is: **{prediction}**")
st.markdown('</div>', unsafe_allow_html=True)

# Get the SPY options data and display the correct option based on the prediction
options_data = get_spy_options()

# Display the recommended option (call or put) based on the prediction
st.markdown('<div class="options-box">', unsafe_allow_html=True)
if prediction == "UP":
    st.write(f"**Recommended SPY Call Option (Expiration: {options_data['expiration_date']}):**")
    st.write(f"- Call Option Strike Price: **${options_data['call_option']}**")
else:
    st.write(f"**Recommended SPY Put Option (Expiration: {options_data['expiration_date']}):**")
    st.write(f"- Put Option Strike Price: **${options_data['put_option']}**")
st.markdown('</div>', unsafe_allow_html=True)

# Footer with "Made by Shriyan Kandula"
st.markdown('<p class="footer">Made by Shriyan Kandula</p>', unsafe_allow_html=True)
