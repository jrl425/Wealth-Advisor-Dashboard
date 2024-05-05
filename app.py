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

# Utility function calculation
# Calculate the mean returns for each asset (column)
returns_mean = returns_data.mean().values

# Ensure the number of columns in returns_data matches the dimensions of the covariance matrix
if len(returns_data.columns) != covariance_matrix.shape[0] or len(returns_data.columns) != covariance_matrix.shape[1]:
    st.error("Mismatch in number of assets and dimensions of covariance matrix")
else:
    # Create example weights assuming equal investment in each asset
    example_weights = np.array([1 / len(returns_data.columns)] * len(returns_data.columns))

    # Calculate portfolio return as a dot product of weights and mean returns
    portfolio_return = np.dot(example_weights, returns_mean)

    # Calculate portfolio variance as a double dot product involving the covariance matrix
    portfolio_variance = np.dot(example_weights, np.dot(covariance_matrix.values, example_weights))

    # Calculate utility as expected return minus half the product of risk aversion coefficient and variance
    utility = portfolio_return - 0.5 * risk_aversion * portfolio_variance

    # Display the results
    st.write(f"Calculated Utility: {utility}")
    st.write(f"Expected Return: {portfolio_return}")
    st.write(f"Variance: {portfolio_variance}")
