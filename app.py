import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Title of the dashboard
st.title("Portfolio Optimization Dashboard")

# Upload CSV file
uploaded_file = st.file_uploader("index_data.csv", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    # Assuming the columns are named 'Expected_Annual_Return' and 'Variance'
    expected_returns = data['Expected_Annual_Return']
    variances = data['Variance']
    tickers = data['Ticker']

    # Risk aversion level (user input)
    risk_aversion = st.slider("Select Risk Aversion Level", 1, 5, 3)

    # Portfolio metrics function for multiple stocks
    def portfolio_metrics_from_df(weights):
        E_R = np.dot(weights, expected_returns)
        variance = np.dot(weights**2, variances)
        return E_R, variance

    # Utility function to maximize
    def negative_utility_from_df(weights):
        E_R, variance = portfolio_metrics_from_df(weights)
        utility = E_R - (risk_aversion / 2) * variance
        return -utility  # Minimize the negative utility to maximize actual utility

    # Constraints and bounds
    constraints_df = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds_df = tuple((0, 1) for _ in tickers)

    # Initial weights evenly distributed
    initial_weights_df = np.array([1/len(tickers)] * len(tickers))

    # Optimization to maximize the utility function
    result_df = minimize(negative_utility_from_df, initial_weights_df, method='SLSQP', bounds=bounds_df, constraints=constraints_df)

    # Display optimized weights and utility
    st.write("Optimized Weights:", dict(zip(tickers, result_df.x)))
    st.write("Maximum Utility:", -result_df.fun)

    # Plotting the results
    fig, ax = plt.subplots()
    E_R, variance = portfolio_metrics_from_df(result_df.x)
    ax.scatter(np.sqrt(variances), expected_returns, label='Individual Stocks')
    ax.scatter(np.sqrt(variance), E_R, color='red', label='Optimized Portfolio')
    ax.set_xlabel('Volatility (Standard Deviation)')
    ax.set_ylabel('Expected Returns')
    ax.legend()
    st.pyplot(fig)
