import streamlit as st
import yfinance as yf
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Function to fetch SPY stock data
def get_stock_data():
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=365 * 2)  # 2 years of data
    data = yf.download('SPY', start=start_date, end=today)
    return data

# Function to calculate technical indicators (SMA50, SMA200, RSI)
def calculate_indicators(data):
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["SMA_200"] = data["Close"].rolling(window=200).mean()
    data["Price Change"] = data["Close"].pct_change()
    
    # RSI Calculation
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    data.dropna(inplace=True)
    return data

# Function to prepare the dataset for training
def prepare_data(data):
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)  # 1 if price goes up, else 0
    
    # Features: SMA, RSI, and Price Change
    features = ["SMA_50", "SMA_200", "RSI", "Price Change"]
    X = data[features]
    y = data["Target"]
    
    return X, y

# Train a Logistic Regression model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    accuracy = model.score(X_test_scaled, y_test)

    return model, scaler, accuracy

# Function to predict market movement
def predict_market(model, scaler, latest_data):
    latest_scaled = scaler.transform([latest_data])
    prediction = model.predict(latest_scaled)[0]
    return "UP" if prediction == 1 else "DOWN"

# Function to recommend an options trade
def get_options_recommendation(prediction):
    return "Call Option" if prediction == "UP" else "Put Option"

# Load and process stock data
data = get_stock_data()
data = calculate_indicators(data)

# Prepare data and train the model
X, y = prepare_data(data)
model, scaler, accuracy = train_model(X, y)

# Streamlit UI
st.title("S&P 500 (SPY) Prediction App")
st.write(f"Model Accuracy: **{accuracy:.2%}**")

st.write("Click the button to get a prediction on whether SPY will go **up or down** for the next trading day.")

# Button to trigger prediction
if st.button("Predict Market Direction"):
    latest_data = data.iloc[-1][["SMA_50", "SMA_200", "RSI", "Price Change"]].values
    market_prediction = predict_market(model, scaler, latest_data)
    option_recommendation = get_options_recommendation(market_prediction)

    st.subheader(f"📈 Market Prediction: **{market_prediction}**")
    st.subheader(f"📊 Options Recommendation: **{option_recommendation}**")

st.markdown("---")
st.markdown("<p style='text-align:center; font-size:14px;'>Made by Shriyan Kandula</p>", unsafe_allow_html=True)
