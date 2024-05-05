import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objs as go

# Title and file setup
st.title("Portfolio Optimization Dashboard")
data = pd.read_csv('inputs/index_data.csv')
covariance_matrix = pd.read_csv('inputs/cov_mat.csv').to_numpy()  # Convert DataFrame to NumPy array directly

# Read the risk-free rate from a text file
try:
    with open('inputs/risk_free_rate.txt', 'r') as file:
        risk_free_rate = float(file.read().strip())
    st.write("Risk-Free Rate Loaded:", risk_free_rate)
except Exception as e:
    st.error(f"Failed to load the risk-free rate: {e}")
    st.stop()

# Assuming columns are named 'Expected_Annual_Return'
expected_returns = data['Expected_Annual_Return']
tickers = data['Ticker']

# Sidebar for risk aversion
with st.sidebar:
    st.header("Risk Survey")
    risk_aversion = st.slider("Select Risk Aversion Level", 1, 5, 3)

# Functions to calculate portfolio metrics and utility
def portfolio_metrics(weights):
    E_R = np.dot(weights, expected_returns)
    variance = np.dot(weights, np.dot(covariance_matrix, weights))  # Properly use the NumPy array
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

# Calculate returns and volatility for the optimized portfolio
opt_return, opt_volatility = portfolio_metrics(optimal_weights)

# Plotting the efficient frontier with Plotly
fig = go.Figure()
# Tangency Portfolio point
fig.add_trace(go.Scatter(
    x=[np.sqrt(opt_volatility)], 
    y=[opt_return], 
    mode='markers',
    marker=dict(size=10, color='red'),
    name='Optimized Portfolio',
    text=[f'Std Dev: {np.sqrt(opt_volatility):.2f}, Expected Return: {opt_return:.2%}'],
    hoverinfo='text+x+y'
))

# Line from risk-free rate to optimized portfolio (Capital Market Line)
fig.add_trace(go.Scatter(
    x=[0, np.sqrt(opt_volatility)],
    y=[risk_free_rate, opt_return],
    mode='lines',
    line=dict(color='black', dash='dash'),
    name='Capital Market Line'
))

# Add risk-free rate point
fig.add_trace(go.Scatter(
    x=[0],
    y=[risk_free_rate],
    mode='markers',
    marker=dict(size=10, color='green'),
    name='Risk-Free Rate'
))

fig.update_layout(
    title='Efficient Frontier with Tangency Portfolio',
    xaxis_title='Volatility (Standard Deviation)',
    yaxis_title='Expected Returns',
    legend_title='Legend',
    hovermode='closest'
)

st.plotly_chart(fig, use_container_width=True)
