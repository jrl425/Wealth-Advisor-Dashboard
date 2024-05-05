import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Load the ticker returns data
returns_data = pd.read_csv('inputs/index_data.csv')
returns_mean = returns_data.mean().values  # Calculate mean returns

# Load the covariance matrix data
covariance_matrix = pd.read_csv('inputs/cov_mat.csv').values  # Ensure this matches the returns data

# Set the risk aversion level (e.g., from a user input or fixed)
risk_aversion = 3

# Define the utility function to minimize (negative utility for maximization)
def utility(weights):
    portfolio_return = np.dot(weights, returns_mean)
    portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
    return -(portfolio_return - 0.5 * risk_aversion * portfolio_variance)  # Negative for maximization

# Set initial guess and bounds
num_assets = len(returns_data.columns)
initial_guess = np.ones(num_assets) / num_assets
bounds = tuple((0, 0.25) for asset in range(num_assets))  # Weights between 0 and 25%

# Constraint to ensure the sum of weights equals 1
constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

# Optimization using SLSQP method
result = minimize(utility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

# Output the results
if result.success:
    optimal_weights = result.x
    st.write("Optimal Weights:", optimal_weights)
else:
    st.error("Optimization did not converge: " + result.message)
