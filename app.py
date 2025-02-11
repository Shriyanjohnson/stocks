import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ta import add_all_ta_features

# Download stock data
def load_data(symbol='SPY'):
    data = yf.download(symbol, start="2018-01-01", end="2025-01-01")
    return data

# Add more indicators using 'ta' library
def add_technical_indicators(data):
    data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume")
    return data

# Prepare data for prediction
def prepare_data(data):
    # Adding technical indicators
    data = add_technical_indicators(data)

    # Drop rows with NaN values
    data.dropna(inplace=True)

    # Feature columns
    features = ['trend_sma_fast', 'trend_sma_slow', 'momentum_rsi', 'momentum_stoch_rsi', 'volatility_bbh', 'volatility_bbl', 'volume_adi']
    
    # Target column (1 if price goes up, 0 if it goes down)
    data['target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

    # Split data into features and target
    X = data[features]
    y = data['target']

    return X, y

# Train the model (XGBoost)
def train_model(X, y):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the XGBoost model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train_scaled, y_train)

    # Evaluate the model
    accuracy = model.score(X_test_scaled, y_test)
    return model, accuracy

# Make prediction
def make_prediction(model, data):
    # Prepare the data
    X = data[['trend_sma_fast', 'trend_sma_slow', 'momentum_rsi', 'momentum_stoch_rsi', 'volatility_bbh', 'volatility_bbl', 'volume_adi']].tail(1)
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Make prediction
    prediction = model.predict(X_scaled)
    return prediction[0]

# Displaying the UI
def show_ui():
    st.title("SPY Stock Prediction & Options Recommendation")
    
    # Load data and model
    data = load_data()
    X, y = prepare_data(data)
    model, accuracy = train_model(X, y)
    
    # Display Model Accuracy
    st.subheader(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Predict next week's movement
    prediction = make_prediction(model, data)
    
    if prediction == 1:
        st.write("### Prediction: Bullish")
        st.write("Recommended Action: **Buy Call Option**")
    else:
        st.write("### Prediction: Bearish")
        st.write("Recommended Action: **Buy Put Option**")
    
    # Display chart for the last 5 years
    st.subheader("SPY Stock Chart (Last 5 Years)")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,
                    open=data['Open'], high=data['High'],
                    low=data['Low'], close=data['Close'],
                    name='SPY'))
    st.plotly_chart(fig)

    # Footer
    st.markdown('<p style="text-align: center;">Made by Shriyan Kandula</p>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    show_ui()
2. Add the UI Features:
We can make it colorful and interactive with Streamlit widgets:

Buttons: You can add buttons that display the prediction or show a graph with one click.
Colors: Add some basic CSS to improve the visual appeal.
Example:

python
Copy
Edit
# Add CSS for the footer and some styling
st.markdown(
    """
    <style>
    .footer {
        text-align: center;
        font-size: 18px;
        color: #4CAF50;
        font-weight: bold;
    }
    .header {
        text-align: center;
        font-size: 36px;
        color: #FF5733;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True
)

# Add button to show prediction
if st.button("Predict SPY Movement"):
    show_ui()
