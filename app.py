import yfinance as yf
import pandas as pd
import numpy as np
import talib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Load SPY data
def load_data():
    ticker = "SPY"
    data = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today").strftime('%Y-%m-%d'))
    return data

# Add technical indicators
def add_technical_indicators(data):
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    macd, macdsignal, macdhist = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data['MACD'] = macd
    data['MACD_Signal'] = macdsignal
    upper, middle, lower = talib.BBANDS(data['Close'], timeperiod=20)
    data['Bollinger_Upper'] = upper
    data['Bollinger_Lower'] = lower
    return data

# Prepare features and target for model training
def prepare_data(data):
    data = data.dropna()  # Drop rows with missing values
    features = ['SMA_50', 'SMA_200', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_Upper', 'Bollinger_Lower', 'Close']
    X = data[features]
    data['Price_Change'] = data['Close'].shift(-1) - data['Close']
    data['Target'] = np.where(data['Price_Change'] > 0, 1, 0)
    y = data['Target']
    return X, y

# Train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler

# Predict the movement
def predict_movement(model, data, scaler):
    data = add_technical_indicators(data)
    last_data = data.iloc[-1:]
    X_latest = last_data[['SMA_50', 'SMA_200', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_Upper', 'Bollinger_Lower', 'Close']]
    X_latest_scaled = scaler.transform(X_latest)
    prediction = model.predict(X_latest_scaled)
    return "UP" if prediction == 1 else "DOWN"

# Function to display options (call or put) based on prediction
def display_option(prediction):
    if prediction == "UP":
        st.write("**Recommended SPY Call Option**")
        st.write("Strike Price: **$425**")  # Example, replace with actual strike price
    else:
        st.write("**Recommended SPY Put Option**")
        st.write("Strike Price: **$420**")  # Example, replace with actual strike price

# Streamlit Layout
st.markdown('<h1 style="text-align:center;">SPY ETF Stock Prediction</h1>', unsafe_allow_html=True)

# Load data and prepare model
data = load_data()
X, y = prepare_data(data)
model, scaler = train_model(X, y)

# Show model accuracy
st.write(f"Model Training Complete. Accuracy: {model.score(X, y) * 100:.2f}%")

# Predict the movement for the week
prediction = predict_movement(model, data, scaler)

st.write(f"**Prediction for SPY ETF movement for the week:** {prediction}")
display_option(prediction)

st.markdown('<footer style="text-align:center;">Made by Shriyan Kandula</footer>', unsafe_allow_html=True)
