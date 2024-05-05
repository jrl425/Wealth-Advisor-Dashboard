import streamlit as st
import pandas as pd
import numpy as np

# Load the ticker returns data
returns_data = pd.read_csv('inputs/index_data.csv')
st.write("Ticker Returns Data Loaded:")
st.dataframe(returns_data.head())

# Calculate expected returns for each ticker if it's a column in returns_data
# Adjust column name as necessary
expected_returns = returns_data.mean().values  # This ensures it's in array form

# Load the covariance matrix data
covariance_matrix = pd.read_csv('inputs/cov_mat.csv').values  # Ensure it's a numpy array
st.write("\nCovariance Matrix Data Loaded:")
st.dataframe(covariance_matrix)

# Initialize the sidebar
st.sidebar.header("User Inputs for Wealth Management")
st.sidebar.subheader("Risk Aversion Survey")
risk_aversion = st.sidebar.slider("Select your risk aversion level:", 1, 5, 3)
st.sidebar.subheader("Investment Details")
investment_amount = st.sidebar.number_input("Enter the amount you want to invest:", min_value=1000, step=1000)

# Define the utility function
def utility(weights):
    portfolio_return = np.dot(weights, expected_returns)  # Use dot for 1D array
    portfolio_risk = np.dot(weights.T, np.dot(covariance_matrix, weights))
    return portfolio_return - 0.5 * risk_aversion * portfolio_risk

# Example weights (equal allocation for simplicity)
weights = np.array([1/len(expected_returns)]*len(expected_returns))

# Calculate and display utility
calculated_utility = utility(weights)
st.write(f"Calculated Utility: {calculated_utility}")
