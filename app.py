import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize

####################################
#Code loads in data
# Load the ticker returns data
returns_data = pd.read_csv('inputs/index_data.csv')
st.write("Ticker Returns Data Loaded:")
st.dataframe(returns_data.head())

# Load the covariance matrix data
covariance_matrix = pd.read_csv('inputs/cov_mat.csv')
st.write("\nCovariance Matrix Data Loaded:")
#covariance_matrix.set_index('Ticker', inplace=True)
st.dataframe(covariance_matrix.head())

#####################################


#####################################
#Side bar code
# Initialize the sidebar
st.sidebar.header("User Inputs for Wealth Management")
st.sidebar.subheader("Risk Aversion Survey")
# Risk aversion input from 1 to 5
risk_aversion = st.sidebar.slider("Select your risk aversion level:", 1, 5, 3)
st.sidebar.subheader("Investment Details")
# Input for investment amount
investment_amount = st.sidebar.number_input("Enter the amount you want to invest:", min_value=1000, step=1000)
#####################################

######################################
#Utility function calculation
def calculate_utility(returns, weights, cov_matrix, risk_aversion):
    """
    Calculate the utility of a portfolio based on risk aversion.

    Parameters:
    - returns: np.array of expected returns for each asset.
    - weights: np.array of portfolio weights for each asset.
    - cov_matrix: DataFrame or np.array of covariance matrix of asset returns.
    - risk_aversion: float, the risk aversion coefficient.

    Returns:
    - Utility value of the portfolio.
    """
    # Calculate expected portfolio return
    portfolio_return = np.dot(weights, returns)
    
    # Calculate portfolio variance
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    
    # Calculate utility
    utility = portfolio_return - 0.5 * risk_aversion * portfolio_variance
    return utility


##########################################


