import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from keras.models import load_model
# Define start and end dates for the data download
start = "2010-01-01"
end = "2022-01-31"

st.title('Stock Trend Prediction')

# Get user input for the stock symbol
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

if user_input:
    # Download data for the stock symbol
    df = yf.download(user_input, start=start, end=end)

    # Describing Data
    st.subheader('Data from 2010-2022')
    st.write(df.describe())

    # Visualizations
    st.subheader('Closing Price vs Time Chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time Chart with 100MA')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100)
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time Chart with 100MA & 200 MA')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100, 'r')
    plt.plot(ma200, 'g')
    plt.plot(df.Close, 'b')
    st.pyplot(fig)

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import TimeSeriesSplit

    df1 = df.copy()
    # Split the data into training and testing sets using TimeSeriesSplit
    X = df1[['Close']]
    y = df1['Close']
    tscv = TimeSeriesSplit(n_splits=5)

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Use X_train, y_train for training and X_test, y_test for testing

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    data_training_array = scaler.fit_transform(X_train)

    # Load my model
    model = load_model('keras_model.h5')

    # Testing Part
    past_100_days = X_train.tail(100)
    final_df = past_100_days.append(X_test, ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100: i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)
    scaler = scaler.scale_
    # My Scale will not be the same scaler for every stock that's why make scaler at 0
    scale_factor = 1/scaler[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    # Final graph
    st.subheader('Prediction vs Original')
    fig2 = plt.figure(figsize=(12, 6))  # adding fig2
    plt.plot(y_test, 'b', label='Original Price')
    plt.plot(y_predicted, 'r', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

else:
    st.warning('Please enter a stock symbol.')
