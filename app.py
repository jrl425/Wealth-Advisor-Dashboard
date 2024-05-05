import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load the ticker returns data
returns_data = pd.read_csv('inputs/index_data.csv')
st.write("Ticker Returns Data Loaded:")
st.dataframe(returns_data.head())

# Ensure that data for expected returns is numeric and deal with non-numeric data appropriately
# Example assuming the column 'Total Expected Return (%)' contains expected returns
if 'Total Expected Return (%)' in returns_data.columns:
    returns_data['Total Expected Return (%)'] = pd.to_numeric(returns_data['Total Expected Return (%)'], errors='coerce')
    expected_returns = returns_data['Total Expected Return (%)'].dropna().values
else:
    st.error('Expected return data column not found in the dataset.')

# Load the covariance matrix data
covariance_matrix = pd.read_csv('inputs/cov_mat.csv').values  # Ensure it's a numpy array
st.write("\nCovariance Matrix Data Loaded:")
st.dataframe(covariance_matrix)

# Initialize the sidebar
st.sidebar.header("User Inputs for Wealth Management")
st.sidebar.subheader("Risk Aversion Survey")
risk_aversion = st.sidebar.slider("Select your risk aversion level:", 1, 10, 3)
st.sidebar.subheader("Investment Details")
investment_amount = st.sidebar.number_input("Enter the amount you want to invest:", min_value=1000, step=1000)

# Define the portfolio performance and utility functions
def portfolio_performance(weights, returns, covariance_matrix):
    port_return = np.dot(weights, returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    return port_return, port_volatility

def minimize_function(weights):
    port_return, port_volatility = portfolio_performance(weights, expected_returns, covariance_matrix)
    utility = port_return - (risk_aversion / 2) * (port_volatility**2)
    return -utility

# Optimization
initial_guess = np.full(len(expected_returns), 1/len(expected_returns))
bounds = tuple((0, 1) for _ in expected_returns)
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

result = minimize(minimize_function, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

if result.success:
    optimal_weights = result.x
    expected_return, volatility = portfolio_performance(optimal_weights, expected_returns, covariance_matrix)
    st.write(f"Optimal Portfolio Weights: {optimal_weights}")
    st.write(f"Expected Return: {expected_return}")
    st.write(f"Volatility: {volatility}")
else:
    st.error("Optimization failed.")
