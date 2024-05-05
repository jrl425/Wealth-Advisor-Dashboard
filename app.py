import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import streamlit as st

# Load the ticker returns data
df = pd.read_csv('inputs/index_data.csv')
st.write("Ticker Returns Data Loaded:")
st.dataframe(returns_data.head())

risk_free_return = 0.05209
risk_free_volatility = 0.0

# Initialize the sidebar
st.sidebar.header("User Inputs for Wealth Management")
st.sidebar.subheader("Risk Aversion Survey")
risk_aversion = st.sidebar.slider("Select your risk aversion level:", 1, 10, 3)
st.sidebar.subheader("Investment Details")
investment_amount = st.sidebar.number_input("Enter the amount you want to invest:", min_value=1000, step=1000)

# Convert annualized standard deviation to covariance matrix
covariance_matrix = np.diag(df['Annualized_Std']**2)
extended_cov_matrix = np.pad(covariance_matrix, ((0, 1), (0, 1)), mode='constant', constant_values=0)

# Extended returns array including the risk-free asset
extended_returns = np.append(df['Total_Expected_Return'].values, risk_free_return)

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
optimal_portfolios = []
for aversion in risk_aversions:
    result = minimize(minimize_function, initial_guess, args=(aversion, extended_returns, extended_cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        port_return, port_volatility = portfolio_performance(result.x, extended_returns, extended_cov_matrix)
        optimal_portfolios.append({
            "Risk Aversion": aversion,
            "Weights": result.x,
            "Expected Return": port_return,
            "Volatility": port_volatility
        })

# Output results
optimal_portfolios_df = pd.DataFrame(optimal_portfolios)

# Plotting the results
plt.figure(figsize=(10, 6))
for index, row in optimal_portfolios_df.iterrows():
    plt.scatter(row["Volatility"], row["Expected Return"], label=f'Risk Aversion {row["Risk Aversion"]}')
plt.xlabel("Volatility (Standard Deviation)")
plt.ylabel("Expected Return")
plt.title("Optimal Portfolio Including Risk-Free Asset: Expected Return vs. Volatility")
plt.legend()
plt.grid(True)
plt.show()

# Print portfolio weights
for index, row in optimal_portfolios_df.iterrows():
    print(f"\nRisk Aversion {row['Risk Aversion']}:")
    print("Portfolio Weights:")
    for i, ticker in enumerate(df['Ticker'].tolist() + ['Risk-Free Asset']):
        print(f"{ticker}: {row['Weights'][i]*100:.2f}%")
    print(f"Expected Return: {row['Expected Return']*100:.2f}%")
    print(f"Volatility: {row['Volatility']*100:.2f}%")
