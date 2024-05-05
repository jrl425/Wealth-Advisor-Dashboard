import streamlit as st
import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import objective_functions
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

# Function to create and optimize portfolio
def optimize_portfolio(returns, covariance_matrix, risk_free_rate, use_sharpe=True):
    # Initialize Efficient Frontier with regularization
    ef = EfficientFrontier(returns, covariance_matrix, weight_bounds=(0, 1))
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)  # Regularization
    
    try:
        if use_sharpe:
            weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        else:
            weights = ef.min_volatility()
    except Exception as e:
        st.write(f"Optimization failed with {('max_sharpe' if use_sharpe else 'min_volatility')}, retrying: {str(e)}")
        # Retry with min_volatility if max_sharpe fails
        if use_sharpe:
            return optimize_portfolio(returns, covariance_matrix, risk_free_rate, use_sharpe=False)
        else:
            raise RuntimeError("Both optimization attempts failed.")
    
    return ef, weights

# Attempt to optimize using max_sharpe, fall back to min_volatility if necessary
efficient_frontier, portfolio_weights = optimize_portfolio(returns, covariance_matrix, 0.05209)

cleaned_weights = efficient_frontier.clean_weights()

# Calculate performance metrics
performance = efficient_frontier.portfolio_performance(verbose=True, risk_free_rate=0.05209)

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
