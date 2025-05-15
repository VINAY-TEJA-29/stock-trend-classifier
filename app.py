import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Trend Classifier", layout="wide")
st.title("📈 Stock Price Trend Classifier")

# User inputs
ticker = st.text_input("Enter stock ticker (e.g. AAPL, GOOGL)", value="AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))

if st.button("Train Model"):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            st.error("No data found for the given ticker and date range.")
            st.stop()
        st.write("Data preview:", df.tail())
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

    # Plots
    st.subheader("📊 Stock Closing Price")
    st.line_chart(df['Close'])

    st.subheader("📉 Stock Volume")
    st.line_chart(df['Volume'])

    # Feature engineering
    df['Return'] = df['Close'].pct_change()
    df['Target'] = np.where(df['Return'].shift(-1) > 0, 1, 0)
    df['5_day_MA'] = df['Close'].rolling(window=5).mean()
    df['10_day_MA'] = df['Close'].rolling(window=10).mean()
    df['Volatility'] = df['Close'].rolling(window=5).std()
    df.dropna(inplace=True)

    features = ['Close', 'Volume', '5_day_MA', '10_day_MA', 'Volatility']
    X = df[features]
    y = df['Target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    st.subheader(f"✅ Model Accuracy: {acc:.2f}")

    # Confusion matrix
    st.subheader("📌 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues")
    st.pyplot(fig)

    # Feature importance
    importances = model.feature_importances_
    feature_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_df = feature_df.sort_values(by='Importance', ascending=False)

    st.subheader("🔍 Feature Importance")
    st.bar_chart(feature_df.set_index('Feature'))

    # Predict next trend
    latest_features = scaler.transform([X.iloc[-1]])
    latest_prediction = model.predict(latest_features)[0]
    trend = "⬆️ Up" if latest_prediction == 1 else "⬇️ Down"

    st.subheader("📅 Next Day Prediction")
    st.success(f"Predicted next trend: {trend}")

    # Download processed data
    csv = df.to_csv().encode('utf-8')
    st.download_button("📥 Download Processed Data", csv, "stock_data.csv", "text/csv")

    # Live price header
    st.subheader("📡 Live Stock Price (updates every second)")

# Live price feed
if st.button("Start Live Price Feed"):
    if ticker.strip() == "":
        st.error("Please enter a valid ticker symbol.")
    else:
        live_placeholder = st.empty()
        for _ in range(60):  # Update for 60 seconds
            live_data = yf.download(tickers=ticker, period="1d", interval="1m")
            if not live_data.empty:
                current_price = live_data['Close'][-1]
                current_time = pd.to_datetime(live_data.index[-1]).strftime("%H:%M:%S")
                live_placeholder.metric(label=f"{ticker} @ {current_time}", value=f"${current_price:.2f}")
            else:
                live_placeholder.error("No live data available.")
            time.sleep(1)
