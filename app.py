import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

####################################
# Code loads in data
# Load the ticker returns data
returns_data = pd.read_csv('inputs/index_data.csv')
st.write("Ticker Returns Data Loaded:")
st.dataframe(returns_data.head())

# Load the covariance matrix data
covariance_matrix = pd.read_csv('inputs/cov_mat.csv')
st.write("\nCovariance Matrix Data Loaded:")
st.dataframe(covariance_matrix.head())

#####################################
# Side bar code
# Initialize the sidebar
st.sidebar.header("User Inputs for Wealth Management")
st.sidebar.subheader("Risk Aversion Survey")
# Risk aversion input from 1 to 5
risk_aversion = st.sidebar.slider("Select your risk aversion level:", 1, 5, 3)
st.sidebar.subheader("Investment Details")
# Input for investment amount
investment_amount = st.sidebar.number_input("Enter the amount you want to invest:", min_value=1000, step=1000)

######################################
# Utility function calculation
def calculate_utility(returns, weights, cov_matrix, risk_aversion):
    # Calculate expected portfolio return and variance
    portfolio_return = sum(weights[i] * returns[i] for i in range(len(returns)))
    portfolio_variance = sum(weights[i] * weights[j] * cov_matrix[i][j] for i in range(len(returns)) for j in range(len(returns)))

    # Calculate utility
    utility = portfolio_return - 0.5 * risk_aversion * portfolio_variance
    return utility

######################################
