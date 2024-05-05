import streamlit as st
import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
import plotly.graph_objects as go

# Assuming 'prices' DataFrame contains the historical price data of assets
# If you don't have this DataFrame, you'll need to obtain or simulate it
# Load your actual data here
# prices = pd.read_csv('path_to_prices.csv')

# Load index data which might be used to validate or further process data (optional)
index_data = pd.read_csv('inputs/index_data.csv')

# Streamlit page configuration
st.set_page_config(page_title="Wealth Management Dashboard", layout="wide")

# Sidebar for user inputs
with st.sidebar:
    st.title("User Preferences")
    risk_aversion = st.slider("Select your risk aversion (1 to 5 scale)", 1, 5, 3)
    leverage = st.slider("Maximum leverage you are willing to take on", 1, 10, 1)

# Calculate expected returns and the covariance matrix using shrinkage
mu = mean_historical_return(prices)  # Annualized mean historical return
S = CovarianceShrinkage(prices).ledoit_wolf()  # Shrinkage estimator

# Initialize Efficient Frontier with the shrunk covariance matrix
ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))

# Optimization
try:
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    performance = ef.portfolio_performance(verbose=True)
except Exception as e:
    st.error(f"Optimization failed: {e}")
    st.stop()

# Display results
st.header("Optimized Portfolio Weights")
st.write(cleaned_weights)

st.header("Portfolio Performance")
st.write(f"Expected Annual Return: {performance[0]:.2%}")
st.write(f"Annual Volatility: {performance[1]:.2%}")
st.write(f"Sharpe Ratio: {performance[2]:.2f}")

# Plotting Efficient Frontier
fig = go.Figure()
fig.add_trace(go.Scatter(x=[0, performance[1]], y=[0, performance[0]], mode='lines', name='Capital Market Line'))
fig.add_trace(go.Scatter(x=[performance[1]], y=[performance[0]], mode='markers', name='Optimal Portfolio', marker=dict(color='red', size=10)))
fig.update_layout(title="Efficient Frontier", xaxis_title="Volatility", yaxis_title="Expected Returns", yaxis_tickformat='.1%')
st.plotly_chart(fig, use_container_width=True)
