import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import streamlit as st

# Initialize the sidebar
st.sidebar.header("User Inputs for Wealth Management")
st.sidebar.subheader("Risk Aversion Survey")
# Risk aversion input from 1 to 5
risk_aversion = st.sidebar.slider("Select your risk aversion level:", 1, 5, 3)
st.sidebar.subheader("Investment Details")
# Input for investment amount
investment_amount = st.sidebar.number_input("Enter the amount you want to invest:", min_value=1000, step=1000)

# Display the user inputs
st.write("## User Inputs Summary")
st.write(f"Risk Aversion Level: {risk_aversion}")
st.write(f"Investment Amount: ${investment_amount:.2f}")
