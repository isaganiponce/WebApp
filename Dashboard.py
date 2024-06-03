import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.title("Sales Forecast Dashboard")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the data
    data = pd.read_csv(uploaded_file)

    st.title("Data View")
    st.write(data)

    # Prepare the sales data
    sales_data = data[['Order Date', 'Sales']]
    sales_data['Order Date'] = pd.to_datetime(sales_data['Order Date'])
    df1 = sales_data.set_index('Order Date')
    monthly_sales = df1.resample('M').mean()

    st.header('Monthly Sales')
    st.line_chart(monthly_sales)

    # Function to check stationarity
    def check_stationarity(timeseries):
        result = adfuller(timeseries, autolag='AIC')
        #p_value = result[1]
        #st.write(f'ADF Statistic: {result[0]}')
        #st.write(f'p-value: {p_value}')
        #st.write('Stationary' if p_value < 0.05 else 'Non-stationary')

    check_stationarity(monthly_sales['Sales'])

    # Define SARIMAX model parameters
    p, d, q = 1, 1, 1
    P, D, Q, s = 1, 1, 1, 12

    # Fit the SARIMAX model
    model = SARIMAX(monthly_sales, order=(p, d, q), seasonal_order=(P, D, Q, s))
    results = model.fit()

    # Forecast future sales
    forecast_periods = 12
    forecast = results.get_forecast(steps=forecast_periods)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    # Plot the results
    st.header("Predicted Sales")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(monthly_sales, label='Observed')
    ax.plot(forecast_mean, label='Forecast', color='red')
    ax.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink')
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()

    st.pyplot(fig)
else:
    st.write("Please upload a CSV file to proceed.")

