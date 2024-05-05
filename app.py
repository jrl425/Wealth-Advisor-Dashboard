import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import streamlit as st

# Load the ticker returns data
df = pd.read_csv('index_data.csv')  # Check your path if needed
covariance_matrix = pd.read_csv('cov_mat.csv', index_col=0).to_numpy()
st.write("Ticker Returns Data Loaded:")
st.dataframe(df.head())

risk_free_return = 0.05209
risk_free_volatility = 0.0

# Initialize the sidebar
st.sidebar.header("User Inputs for Wealth Management")
st.sidebar.subheader("Risk Aversion Survey")
risk_aversions = [1, 10, 20, 25, 30, 40, 48]
risk_aversion = st.sidebar.selectbox("Select your risk aversion level:", risk_aversions)
st.sidebar.subheader("Investment Details")
investment_amount = st.sidebar.number_input("Enter the amount you want to invest:", min_value=1000, step=1000)

# Convert annualized standard deviation to covariance matrix if necessary
extended_cov_matrix = np.pad(covariance_matrix, ((0, 1), (0, 1)), mode='constant', constant_values=0)
extended_returns = np.append(df['Total Expected Return (%)'].values, risk_free_return)

# Initial guess and bounds
initial_guess = np.full(len(extended_returns), 1/len(extended_returns))
bounds = tuple((0, 1) for _ in range(len(extended_returns)))

# Constraints
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

# Portfolio performance calculation
def portfolio_performance(weights, returns, covariance_matrix):
    port_return = np.sum(weights * returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    return port_return, port_volatility

# Utility function to minimize
def minimize_function(weights, risk_aversion, returns, covariance_matrix):
    port_return, port_volatility = portfolio_performance(weights, returns, covariance_matrix)
    utility = port_return - (risk_aversion / 2) * (port_volatility**2)
    return -utility

# Optimization process
result = minimize(minimize_function, initial_guess, args=(risk_aversion, extended_returns, extended_cov_matrix),
                  method='SLSQP', bounds=bounds, constraints=constraints)
if result.success:
    port_return, port_volatility = portfolio_performance(result.x, extended_returns, extended_cov_matrix)
    optimal_portfolio = {
        "Risk Aversion": risk_aversion,
        "Weights": result.x,
        "Expected Return": port_return,
        "Volatility": port_volatility
    }

    # Display results
    st.write(f"Optimal Portfolio for Risk Aversion {risk_aversion}:")
    st.write(f"Expected Return: {port_return*100:.2f}%")
    st.write(f"Volatility: {port_volatility*100:.2f}%")
    st.write("Portfolio Weights:")
    for i, ticker in enumerate(df['Ticker'].tolist() + ['Risk-Free Asset']):
        st.write(f"{ticker}: {optimal_portfolio['Weights'][i]*100:.2f}%")

# Plotting the results
fig, ax = plt.subplots()
ax.scatter(optimal_portfolio['Volatility'], optimal_portfolio['Expected Return'], label=f'Risk Aversion {optimal_portfolio["Risk Aversion"]}')
ax.set_xlabel("Volatility (Standard Deviation)")
ax.set_ylabel("Expected Return")
ax.set_title("Optimal Portfolio Including Risk-Free Asset: Expected Return vs. Volatility")
ax.legend()
ax.grid(True)
st.pyplot(fig)
