import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

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
        }
        .header {
            background-color: #4682b4;
            color: white;
            padding: 20px;
            border-radius: 10px;
        }
        .prediction-box {
            background-color: #ffcccb;
            padding: 15px;
            border-radius: 10px;
            color: #8b0000;
            font-size: 20px;
        }
        .data-table {
            border: 1px solid #4682b4;
            border-radius: 10px;
            padding: 10px;
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

# Function to download the S&P 500 data
def load_data():
    ticker = "^GSPC"  # S&P 500 symbol
    today = datetime.today().strftime('%Y-%m-%d')  # Get today's date dynamically
    data = yf.download(ticker, start="2020-01-01", end=today)  # Use dynamic end date
    return data

# Function to make a simple prediction based on the last two closing prices
def simple_prediction(data):
    last_close = data['Close'].iloc[-1]
    previous_close = data['Close'].iloc[-2]
    
    # Convert to float (to make sure we're working with scalar numbers)
    last_close = float(last_close)
    previous_close = float(previous_close)
    
    if last_close > previous_close:
        return "UP"
    else:
        return "DOWN"

# Streamlit layout for the app
st.markdown('<p class="title">S&P 500 Prediction App</p>', unsafe_allow_html=True)
st.write("This simple app predicts if the S&P 500 will go UP or DOWN based on the last two closing prices.")

# Add a refresh button to update the data
if st.button('Refresh Data', key='refresh', help="Click to refresh the data"):
    data = load_data()
    st.write("### Latest Data Refreshed")
    st.write(data.tail())
else:
    data = load_data()

# Display the latest data in a styled table
st.markdown('<p class="header">### Latest S&P 500 Data</p>', unsafe_allow_html=True)
st.markdown('<div class="data-table">', unsafe_allow_html=True)
st.write(data.tail())
st.markdown('</div>', unsafe_allow_html=True)

# Make a simple prediction
prediction = simple_prediction(data)

# Display prediction in a styled box
st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
st.write(f"The prediction for tomorrow's S&P 500 movement is: **{prediction}**")
st.markdown('</div>', unsafe_allow_html=True)

# Footer with "Made by Shriyan Kandula"
st.markdown('<p class="footer">Made by Shriyan Kandula</p>', unsafe_allow_html=True)
