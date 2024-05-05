import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import streamlit as st

####################################
#Code loads in data

inputs/cov_mat.csv

# Load the ticker returns data
returns_data = pd.read_csv('inputs/index_data.csv')
print("Ticker Returns Data Loaded:")
print(returns_data.head())  # Display the first few rows to verify it's loaded correctly

# Load the covariance matrix data
covariance_matrix = pd.read_csv('inputs/cov_mat.csv')
print("\nCovariance Matrix Data Loaded:")
print(covariance_matrix.head())

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



