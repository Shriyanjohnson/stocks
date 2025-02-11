import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import plotly.graph_objs as go

# Load data
def load_data(ticker='SPY'):
    data = yf.download(ticker, period="5y", interval="1d")
    return data

# Prepare data for prediction with simple features
def prepare_data(data):
    # Ensure there is enough data for calculations
    if data.shape[0] < 200:
        st.error("Not enough data to calculate SMA_50 and SMA_200. Please select a different ticker or timeframe.")
        return None, None

    # Calculate moving averages and RSI
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = 100 - (100 / (1 + (data['Close'].diff(1).where(lambda x: x > 0, 0).rolling(window=14).mean() /
                                    data['Close'].diff(1).where(lambda x: x < 0, 0).rolling(window=14).mean())))

    # Ensure all calculated columns are present
    required_columns = ['SMA_50', 'SMA_200', 'RSI']
    
    # Handle missing columns and drop NaN rows if columns exist
    missing_columns = [col for col in required_columns if col not in data.columns or data[col].isna().all()]
    if missing_columns:
        st.error(f"Missing or insufficient data for columns: {missing_columns}.")
        return None, None

    # Drop rows with NaN values in the calculated columns
    data.dropna(subset=required_columns, inplace=True)

    features = ['SMA_50', 'SMA_200', 'RSI']

    # Target column (1 if price goes up, 0 if it goes down)
    data['target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

    X = data[features]
    y = data['target']

    return X, y

# Train the model (Logistic Regression)
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()

    # Handle any NaN values in the dataset by filling with the column mean before scaling
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_test.mean())

    # Scale the data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    accuracy = model.score(X_test_scaled, y_test)
    return model, accuracy

# Make prediction
def make_prediction(model, data):
    X = data[['SMA_50', 'SMA_200', 'RSI']].tail(1)
    
    # Handle any NaN values in the most recent data
    X = X.fillna(X.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    prediction = model.predict(X_scaled)
    return prediction[0]

# Show UI
def show_ui():
    st.title("S&P 500 Prediction and Options Recommendation")
    ticker = st.text_input('Enter Ticker Symbol', 'SPY')

    # Load data
    data = load_data(ticker)
    
    st.write("Stock Data (Last 5 Years):")
    st.write(data.tail())

    # Prepare data
    X, y = prepare_data(data)
    
    if X is None or y is None:
        return  # Exit if data preparation failed

    # Train model
    model, accuracy = train_model(X, y)

    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Show prediction
    prediction = make_prediction(model, data)
    
    if prediction == 1:
        st.write("Prediction: The stock is predicted to go up. Consider buying a call option.")
    else:
        st.write("Prediction: The stock is predicted to go down. Consider buying a put option.")
        
    # Show chart using plotly
    if st.button('Show Chart'):
        fig = go.Figure()

        # Add closing price trace
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))

        fig.update_layout(title=f'{ticker} Closing Price Over Last 5 Years',
                          xaxis_title='Date',
                          yaxis_title='Price (USD)',
                          template="plotly_dark")

        st.plotly_chart(fig)

    st.markdown('<p class="footer">Made by Shriyan Kandula</p>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    show_ui()
