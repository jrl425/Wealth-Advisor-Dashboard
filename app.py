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

# Set up Streamlit layout
st.set_page_config(layout="wide")
st.sidebar.title("Risk Aversion Survey")
risk_aversion = st.sidebar.slider("Select your risk aversion (1 to 5)", 1, 5, 3)

# Ensure returns and covariance matrix are aligned and correctly formatted
returns = index_data.set_index('Ticker')['Expected_Annual_Return']
covariance_matrix = cov_mat.values

# Initialize Efficient Frontier
ef = EfficientFrontier(returns, covariance_matrix, weight_bounds=(0,1))
weights = ef.max_sharpe(risk_free_rate=0.05209)
cleaned_weights = ef.clean_weights()

# Calculate performance metrics
performance = ef.portfolio_performance(verbose=True, risk_free_rate=0.05209)

# Plotting Efficient Frontier
fig = go.Figure()
fig.add_trace(go.Scatter(x=[0, 0.3], y=[0, 0.15], mode='lines', name='Efficient Frontier'))

# Optimal portfolio point
optimal_risk, optimal_return = performance[1], performance[0]
fig.add_trace(go.Scatter(x=[optimal_risk], y=[optimal_return],
                         mode='markers+text', text=["Optimal"], textposition="top center",
                         marker=dict(color='red', size=10), name='Optimal Portfolio'))

# Display the graph
st.plotly_chart(fig, use_container_width=True)

# Display weights and performance metrics
st.write("Portfolio Weights:", cleaned_weights)
st.write("Expected Annual Return:", performance[0])
st.write("Annual Volatility:", performance[1])
st.write("Sharpe Ratio:", performance[2])
