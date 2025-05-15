import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📈 Stock Price Trend Classifier")

ticker = st.text_input("Enter stock ticker (e.g. AAPL, GOOGL)", value="AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))

if st.button("Train Model"):
    df = yf.download(ticker, start=start_date, end=end_date)
    st.write("Data preview:", df.tail())
    st.subheader("📊 Stock Closing Price")
    st.line_chart(df['Close'])

    st.subheader("📉 Stock Volume")
    st.line_chart(df['Volume'])

    

    df['Return'] = df['Close'].pct_change()
    df['Target'] = np.where(df['Return'].shift(-1) > 0, 1, 0)

    df['5_day_MA'] = df['Close'].rolling(window=5).mean()
    df['10_day_MA'] = df['Close'].rolling(window=10).mean()
    df['Volatility'] = df['Close'].rolling(window=5).std()

    df.dropna(inplace=True)

    features = ['Close', 'Volume', '5_day_MA', '10_day_MA', 'Volatility']
    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.subheader(f"Model Accuracy: {acc:.2f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues")
    st.pyplot(fig)
