import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Title of the dashboard
st.title("Portfolio Optimization Dashboard")

# Sidebar for Risk Aversion and Details
with st.sidebar:
    st.header("Risk Survey")
    # Example questions or inputs
    st.write("1. Question 1")
    st.write("2. Question 2")
    st.write("3. Question 3")
    st.write("4. Question 4")
    st.write("5. Question 5")
    
    risk_aversion = st.slider("Select Risk Aversion Level", 1, 5, 3)
    st.header("Details")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    retirement_age = st.number_input("Retirement Age", min_value=18, max_value=100, value=65)
    initial_investment = st.number_input("Initial Investment", min_value=0)
    annual_contribution = st.number_input("Annual Contribution", min_value=0)
    desired_retirement_income = st.number_input("Desired Retirement Income", min_value=0)

# Main content: Portfolio Allocation and Investment Growth
st.header("Portfolio Allocation")
file_path = 'inputs/index_data.csv'
data = pd.read_csv(file_path)

expected_returns = data['Expected_Annual_Return']
variances = data['Variance']
tickers = data['Ticker']

def portfolio_metrics_from_df(weights):
    E_R = np.dot(weights, expected_returns)
    variance = np.dot(weights**2, variances)
    return E_R, variance

def negative_utility_from_df(weights):
    E_R, variance = portfolio_metrics_from_df(weights)
    utility = E_R - (risk_aversion / 2) * variance
    return -utility

constraints_df = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
bounds_df = tuple((0, 1) for _ in tickers)
initial_weights_df = np.array([1/len(tickers)] * len(tickers))

result_df = minimize(negative_utility_from_df, initial_weights_df, method='SLSQP', bounds=bounds_df, constraints=constraints_df)

st.write("Optimized Weights:", dict(zip(tickers, result_df.x)))
st.write("Maximum Utility:", -result_df.fun)

fig, ax = plt.subplots()
E_R, variance = portfolio_metrics_from_df(result_df.x)
ax.scatter(np.sqrt(variances), expected_returns, label='Individual Stocks')
ax.scatter(np.sqrt(variance), E_R, color='red', label='Optimized Portfolio')
ax.set_xlabel('Volatility (Standard Deviation)')
ax.set_ylabel('Expected Returns')
ax.legend()
st.pyplot(fig)

st.header("Investment Growth")
# Example plot for investment growth (placeholder)
fig2, ax2 = plt.subplots()
ages = np.arange(age, retirement_age + 1)
investment_values = initial_investment * (1 + 0.05) ** np.arange(len(ages)) + annual_contribution
ax2.plot(ages, investment_values, color='green')
ax2.set_xlabel('Age')
ax2.set_ylabel('Investment Value')
ax2.title.set_text('Projected Investment Growth Over Time')
st.pyplot(fig2)
