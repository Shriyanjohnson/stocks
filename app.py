import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf

# Function to fetch SPY data
def get_spy_data():
    today = dt.datetime.today().strftime('%Y-%m-%d')
    spy = yf.Ticker("SPY")
    df = spy.history(period="5y")  # Fetch last 5 years of data
    df = df[['Close']]
    df.rename(columns={'Close': 'SPY Price'}, inplace=True)
    return df

# Function to predict market movement (basic logic for demo)
def predict_market():
    data = get_spy_data()
    last_price = data.iloc[-1]['SPY Price']
    
    # Simulated prediction: Random up/down based on last movement
    trend = np.random.choice(["Up", "Down"], p=[0.55, 0.45])  # Slight bias towards up
    
    # Determine option type
    option_type = "Call" if trend == "Up" else "Put"

    # Suggest strike price and expiration (nearest Friday)
    strike_price = round(last_price * (1.02 if option_type == "Call" else 0.98), 2)
    today = dt.datetime.today()
    days_until_friday = (4 - today.weekday()) % 7
    expiration_date = (today + dt.timedelta(days=days_until_friday)).strftime('%Y-%m-%d')

    return trend, option_type, strike_price, expiration_date

# Streamlit UI
st.set_page_config(page_title="SPY Prediction", layout="wide")

st.markdown("<h1 style='text-align: center; color: #FF5733;'>📈 SPY Market Prediction 📉</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🔮 Predict Market Direction"):
        trend, option_type, strike_price, expiration_date = predict_market()

        st.markdown(f"""
        <div style='text-align: center; font-size: 24px; padding: 15px; background-color: #f4f4f4; border-radius: 10px;'>
            <b>📊 Prediction: {trend}</b> <br><br>
            <b>💼 Option Type: {option_type}</b> <br>
            <b>💰 Strike Price: ${strike_price}</b> <br>
            <b>📅 Expiration Date: {expiration_date}</b> <br>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: gray;'>Made by Shriyan Kandula</p>", unsafe_allow_html=True)
