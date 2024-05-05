import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Load the ticker returns data
returns_data = pd.read_csv('inputs/index_data.csv')
st.write("Ticker Returns Data Loaded:")
st.dataframe(returns_data.head())

# Load the covariance matrix data
covariance_matrix = pd.read_csv('inputs/cov_mat.csv')
st.write("\nCovariance Matrix Data Loaded:")
st.dataframe(covariance_matrix.head())

# Portfolio performance calculation
def portfolio_performance(weights, returns, cov_matrix):
    # Ensure the inputs are numpy arrays and have correct shapes
    weights = np.array(weights)
    returns = np.array(returns)
    cov_matrix = np.array(cov_matrix)

    print("Weights Shape:", weights.shape)  # Debugging
    print("Returns Shape:", returns.shape)  # Debugging
    print("Covariance Matrix Shape:", cov_matrix.shape)  # Debugging

    # Calculate expected portfolio return
    portfolio_return = np.dot(weights, returns)
    
    # Calculate portfolio volatility (standard deviation of returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    return portfolio_return, portfolio_volatility

# Utility function calculation
def calculate_utility(returns, weights, cov_matrix, risk_aversion):
    # Calculate expected portfolio return and volatility
    portfolio_return, portfolio_volatility = portfolio_performance(weights, returns, cov_matrix)
    
    # Calculate utility
    utility = portfolio_return - 0.5 * risk_aversion * portfolio_volatility ** 2
    return utility

# Efficient Frontier Calculation
def efficient_frontier(returns, cov_matrix, num_portfolios=100):
    num_assets = len(returns)
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_return, portfolio_volatility = portfolio_performance(weights, returns, cov_matrix)
        results[0, i] = portfolio_volatility
        results[1, i] = portfolio_return
        results[2, i] = calculate_utility(returns, weights, cov_matrix, risk_aversion)

    return results, weights_record

# Example usage
returns = returns_data.iloc[0, 1:].values  # Ensure correct indexing and convert to numpy array
cov_matrix = covariance_matrix.values  # Ensure the covariance matrix is a numpy array
results, weights = efficient_frontier(returns, cov_matrix)

# Plotting the Efficient Frontier
def plot_efficient_frontier(results):
    plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis')
    plt.title('Efficient Frontier')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Expected Returns')
    plt.colorbar(label='Utility')
    plt.show()

plot_efficient_frontier(results)
