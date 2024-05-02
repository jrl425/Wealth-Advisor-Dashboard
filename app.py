import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Title and file setup
st.title("Portfolio Optimization Dashboard")
file_path = 'inputs/index_data.csv'
data = pd.read_csv(file_path)

# Assuming columns are named 'Expected_Annual_Return' and 'Variance'
expected_returns = data['Expected_Annual_Return']
variances = data['Variance']
tickers = data['Ticker']

# Sidebar for risk aversion
with st.sidebar:
    st.header("Risk Survey")
    # Example risk aversion input
    risk_aversion = st.slider("Select Risk Aversion Level", 1, 5, 3)

# Functions to calculate portfolio metrics and utility
def portfolio_metrics(weights):
    E_R = np.dot(weights, expected_returns)
    variance = np.dot(weights, np.dot(np.diag(variances), weights))
    return E_R, variance

def negative_utility(weights):
    E_R, variance = portfolio_metrics(weights)
    utility = E_R - (risk_aversion / 2) * variance
    return -utility

# Optimizer setup
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
bounds = tuple((0, 1) for _ in tickers)
initial_weights = np.array([1 / len(tickers)] * len(tickers))

# Optimize portfolio
result = minimize(negative_utility, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = result.x

# Display optimized weights (converted to percentages)
st.write("Optimized Weights:", dict(zip(tickers, optimal_weights * 100)))
st.write("Maximum Utility:", -result.fun)

# Generate portfolios for the efficient frontier
n_portfolios = 100
all_weights = np.random.dirichlet(np.ones(len(tickers)), n_portfolios)
returns = [portfolio_metrics(weights)[0] for weights in all_weights]
volatilities = [np.sqrt(portfolio_metrics(weights)[1]) for weights in all_weights]

# Plotting the efficient frontier
fig, ax = plt.subplots()
ax.scatter(volatilities, returns, c='blue', label='Possible Portfolios')
opt_return, opt_volatility = portfolio_metrics(optimal_weights)
ax.scatter(opt_volatility, opt_return, color='red', label='Optimized Portfolio')
ax.set_xlabel('Volatility (Standard Deviation)')
ax.set_ylabel('Expected Returns')
ax.legend()
st.pyplot(fig)
