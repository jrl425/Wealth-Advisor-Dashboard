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
#returns_mean = returns_data['Total Expected Return (%)'].mean()
def utility(weights, returns, cov_matrix, risk_aversion):
    portfolio_return = np.dot(weights, returns.mean())
    portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
    return -(portfolio_return - 0.5 * risk_aversion * portfolio_variance)  # Negative for minimization

# Set initial guess and constraints
num_assets = len(returns_data.columns)
initial_guess = np.ones(num_assets) / num_assets
bounds = tuple((0, 1) for asset in range(num_assets))  # Weights between 0 and 1
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})  # Sum of weights must be 1

# Optimization
result = minimize(utility, initial_guess, args=(returns_data, covariance_matrix.values, risk_aversion),
                  method='SLSQP', bounds=bounds, constraints=constraints)

# Check results
if result.success:
    optimal_weights = result.x
    st.write(f"Optimal Weights: {optimal_weights}")
else:
    st.write("Optimization did not converge")
