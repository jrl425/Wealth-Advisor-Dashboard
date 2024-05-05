import streamlit as st
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
import plotly.graph_objects as go

# Load the returns data and covariance matrix
returns_data = pd.read_csv('inputs/index_data.csv')
cov_matrix = pd.read_csv('inputs/cov_mat.csv', index_col=0)

# Convert annual returns to expected returns and covariance matrix
mu = returns_data['Expected_Annual_Return']
S = cov_matrix

def optimize_portfolio(risk_aversion):
    # Create the Efficient Frontier Object
    ef = EfficientFrontier(mu, S)
    # Maximize the Sharpe ratio, then calculate weights using the utility function
    raw_weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    performance = ef.portfolio_performance(verbose=True)
    return cleaned_weights, performance

# Streamlit interface
st.title('Wealth Management Dashboard')

# User inputs for risk aversion
risk_level = st.slider("Select your risk aversion level (1=Low, 5=High):", 1, 5, 3)

# Show the calculated optimal portfolio
if st.button('Calculate Optimal Portfolio'):
    weights, performance = optimize_portfolio(risk_level)
    st.write("Portfolio Weights:", weights)
    st.write("Expected Annual Return:", performance[0])
    st.write("Annual Volatility:", performance[1])
    st.write("Sharpe Ratio:", performance[2])

    # Visualizing the efficient frontier
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[performance[1]], y=[performance[0]], mode='markers+text', 
                             text=['Optimal Portfolio'], textposition='top center'))
    st.plotly_chart(fig, use_container_width=True)
