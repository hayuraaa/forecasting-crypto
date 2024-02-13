import streamlit as st
import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Streamlit app
def main():
    st.title("Prediksi Crypto Metaverse")
    st.write("Prediksi Menggunakan Model Long Short Term Memory dan Gated Recurrent Unit")
    
    # Sidebar Input Data
    st.sidebar.header("Data Download")
    stock_symbol = st.sidebar.text_input("Masukkan Nama Coin (e.g., SAND-USD):", "SAND-USD")
    start_date = st.sidebar.date_input("Start Date (80% data latih)", pd.to_datetime("2022-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

    # Download stock price data
    data = yf.download(stock_symbol, start=start_date, end=end_date)

    # Proses Data
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Data preparation
    n_steps = 120
    X, y = prepare_data(scaled_data, n_steps)

    # Splitting into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Reshape data for LSTM and GRU models
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    X_train_gru = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_gru = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Sidebar for model selection
    st.sidebar.header("Select Model")
    model_type = st.sidebar.selectbox("Select Model Type:", ["LSTM", "GRU"])
    

    # Mengambil Mode
    if model_type == "LSTM":
        final_model = load_model("stx_model_lstm.h5")
    else:
        final_model = load_model("stx_model_gru.h5")

    # Model evaluation
    y_pred = final_model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Perhitungan Evaluasi
    mse = mean_squared_error(y_test_orig, y_pred)    #  perhitungan MSE
    rmse = math.sqrt(mse)                            #  perhitungan RMSE    
    mad = np.mean(np.abs(y_test_orig - y_pred))      #  perhitungan MAD
    mape = np.mean(np.abs((y_test_orig - y_pred) / y_test_orig)) * 100


    # Display results
    
    st.header(f"Results for {model_type} Model")
    st.write("Mean Squared Error (MSE):", mse)
    st.write("Root Mean Squared Error (RMSE):", rmse)
    st.write("Mean Absolute Deviation (MAD):", mad)
    st.write("Mean Absolute Percentage Error (MAPE):", mape)

    # Visualize predictions
    st.header("Visual Prediksi")
    visualize_predictions(data, train_size, n_steps, y_test_orig, y_pred)
    
    
    # Display combined actual and predicted data table with time information
    st.header("Table Close Harga Asli dan Harga Prediksi")
    
    # Add time information to the header
    st.write("Data range:", data.index[train_size + n_steps:].min(), "to", data.index[train_size + n_steps:].max())
    
    
    # Calculate the difference between actual and predicted prices
    price_difference = y_test_orig.flatten() - y_pred.flatten()
    
    # Calculate the percentage difference
    percentage_difference = (price_difference / y_test_orig.flatten()) * 100
    
    # Convert predicted prices to strings and cut off decimal places after the 5th digit
    predicted_prices_str = [f"{val:.5f}" for val in y_pred.flatten()]

    # Combine data, time information, and price difference into one dataframe with column names
    combined_data = pd.DataFrame({
        'Tanggal': data.index[train_size + n_steps:],
        'Actual_Prices': y_test_orig.flatten(),
        'Predicted_Prices': predicted_prices_str,
        'Price_Difference': abs(price_difference),
        'Percentage_Difference': abs(percentage_difference)
    })
    
     # Format the 'Percentage_Difference' column to include the percentage symbol
    combined_data['Percentage_Difference'] = combined_data['Percentage_Difference'].map("{:.2f}%".format)


    # Display the combined data table
    st.table(combined_data)


def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        lag_values = data[i:(i + n_steps), 0]
        X.append(np.concatenate([lag_values, [data[i + n_steps, 0]]]))
        y.append(data[i + n_steps, 0])
    return np.array(X), np.array(y)


def visualize_predictions(data, train_size, n_steps, y_test_orig, y_pred):
    fig = go.Figure()


    fig.add_trace(go.Scatter(x=data.index[train_size + n_steps:],
                             y=y_test_orig.flatten(),
                             mode='lines',
                             name="Actual Stock Prices",
                             line=dict(color='blue')))

    fig.add_trace(go.Scatter(x=data.index[train_size + n_steps:],
                             y=y_pred.flatten(),
                             mode='lines',
                             name="Predicted Stock Prices",
                             line=dict(color='red')))

    fig.update_layout(title="Stock Price Prediction",
                      xaxis_title="Date",
                      yaxis_title="Stock Price (USD)",
                      template='plotly_dark')
    
        

    st.plotly_chart(fig)
    

     # Plot all prices
    fig_all_prices = go.Figure()

    fig_all_prices.add_trace(go.Scatter(x=data.index, y=data['Open'], mode='lines', name='Opening Price', line=dict(color='red')))
    fig_all_prices.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Closing Price', line=dict(color='green')))
    fig_all_prices.add_trace(go.Scatter(x=data.index, y=data['Low'], mode='lines', name='Low Price', line=dict(color='yellow')))
    fig_all_prices.add_trace(go.Scatter(x=data.index, y=data['High'], mode='lines', name='High Price', line=dict(color='blue')))

    fig_all_prices.update_layout(
        title='Stock Price History',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Stock Price'),
        legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
    )

    # Plot subplots for each individual price
    fig_subplots = make_subplots(rows=2, cols=2, subplot_titles=('Opening Price', 'Closing Price', 'Low Price', 'High Price'))

    fig_subplots.add_trace(go.Scatter(x=data.index, y=data['Open'], mode='lines', name='Opening Price', line=dict(color='red')), row=1, col=1)
    fig_subplots.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Closing Price', line=dict(color='green')), row=1, col=2)
    fig_subplots.add_trace(go.Scatter(x=data.index, y=data['Low'], mode='lines', name='Low Price', line=dict(color='yellow')), row=2, col=1)
    fig_subplots.add_trace(go.Scatter(x=data.index, y=data['High'], mode='lines', name='High Price', line=dict(color='blue')), row=2, col=2)

    fig_subplots.update_layout(title='Stock Price Subplots', showlegend=False)

    st.plotly_chart(fig_all_prices)
    st.plotly_chart(fig_subplots)

    


if __name__ == "__main__":
    main()
