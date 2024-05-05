import streamlit as st
import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
import plotly.graph_objects as go

# Load data
index_data = pd.read_csv('inputs/index_data.csv')
cov_mat = pd.read_csv('inputs/cov_mat.csv')
risk_free_rate = 0.05209  # From your previous inputs or loaded from a file

# Streamlit page configuration
st.set_page_config(page_title="Wealth Management Dashboard", layout="wide")

# Sidebar for user inputs
with st.sidebar:
    st.title("User Preferences")
    risk_aversion = st.slider("Select your risk aversion (1 to 5 scale)", 1, 5, 3)
    leverage = st.slider("Maximum leverage you are willing to take on", 1, 10, 1)
    uploaded_file = st.file_uploader("Upload a custom list of tickers (CSV)")

# Prepare data
returns = index_data.set_index('Ticker')['Expected_Annual_Return']
covariance_matrix = cov_mat.values

# Initialize Efficient Frontier
ef = EfficientFrontier(returns, covariance_matrix, weight_bounds=(0, 1))
ef.add_objective(lambda x: risk_aversion * x.T @ covariance_matrix @ x)  # Adding a custom utility function based on risk aversion

# Optimization
try:
    raw_weights = ef.max_sharpe(risk_free_rate)
    cleaned_weights = ef.clean_weights()
    performance = ef.portfolio_performance(verbose=True, risk_free_rate=risk_free_rate)
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
