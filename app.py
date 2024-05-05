import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Load the ticker returns data
returns_data = pd.read_csv('inputs/index_data.csv')
# Convert all columns to float, handling non-numeric values gracefully
returns_data = returns_data.apply(pd.to_numeric, errors='coerce')
# Fill any NaN values that arise from non-numeric data with the mean of the column
returns_data.fillna(returns_data.mean(), inplace=True)

st.write("Ticker Returns Data Loaded:")
st.dataframe(returns_data.head())

# Load the covariance matrix data
covariance_matrix = pd.read_csv('inputs/cov_mat.csv')
# Ensure covariance matrix is numeric
covariance_matrix = covariance_matrix.apply(pd.to_numeric, errors='coerce')
# Handling NaN by replacing them with zeroes might not be appropriate for a covariance matrix; consider an error or re-evaluation if NaNs exist
covariance_matrix.fillna(0, inplace=True)

st.write("\nCovariance Matrix Data Loaded:")
st.dataframe(covariance_matrix.head())

# Initialize the sidebar
st.sidebar.header("User Inputs for Wealth Management")
st.sidebar.subheader("Risk Aversion Survey")
risk_aversion = st.sidebar.slider("Select your risk aversion level:", 1, 5, 3)
st.sidebar.subheader("Investment Details")
investment_amount = st.sidebar.number_input("Enter the amount you want to invest:", min_value=1000, step=1000)

# Utility function calculation
def calculate_utility(returns, weights, cov_matrix, risk_aversion):
    # Convert inputs to numpy arrays for matrix operations
    weights = np.array(weights)
    returns_mean = returns.mean().values
    cov_matrix = np.array(cov_matrix)

    portfolio_return = np.dot(weights, returns_mean)
    portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
    utility = portfolio_return - 0.5 * risk_aversion * portfolio_variance
    return utility, portfolio_return, portfolio_variance

# Test the utility function with some example weights
example_weights = np.array([0.1] * len(returns_data.columns))
utility, expected_return, variance = calculate_utility(returns_data, example_weights, covariance_matrix.values, risk_aversion)
st.write(f"Calculated Utility: {utility}, Expected Return: {expected_return}, Variance: {variance}")
