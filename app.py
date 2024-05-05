import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Load the ticker returns data
returns_data = pd.read_csv('inputs/index_data.csv')
st.write("Ticker Returns Data Loaded:")
st.dataframe(returns_data.head())

# Load the covariance matrix data
covariance_matrix = pd.read_csv('inputs/cov_mat.csv')
st.write("\nCovariance Matrix Data Loaded:")
st.dataframe(covariance_matrix.head())

# Initialize the sidebar
st.sidebar.header("User Inputs for Wealth Management")
st.sidebar.subheader("Risk Aversion Survey")
risk_aversion = st.sidebar.slider("Select your risk aversion level:", 1, 5, 3)
st.sidebar.subheader("Investment Details")
investment_amount = st.sidebar.number_input("Enter the amount you want to invest:", min_value=1000, step=1000)

# Calculate expected returns and portfolio variance
expected_returns = returns_data.mean()
portfolio_variance = lambda weights: weights.T @ covariance_matrix @ weights

# Define the utility function
def utility(weights):
    portfolio_return = weights.T @ expected_returns
    portfolio_risk = portfolio_variance(weights)
    utility = portfolio_return - 0.5 * risk_aversion * portfolio_risk
    return utility

# Example weights (equal allocation for simplicity)
weights = np.array([1/len(returns_data.columns)]*len(returns_data.columns))

# Calculate and display utility
calculated_utility = utility(weights)
st.write(f"Calculated Utility: {calculated_utility}")
