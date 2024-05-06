import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import plotly.graph_objs as go

# Load the ticker returns data
df = pd.read_csv('inputs/index_data.csv')


st.write("Ticker Returns Data Loaded:")
st.dataframe(df.head())

#10 year treasury
risk_free_return = 0.04497
risk_free_volatility = 0.0

# Sidebar for user inputs
st.sidebar.header("User Inputs for Wealth Management")
st.sidebar.subheader("chinese people rock")
#risk_aversion = st.sidebar.slider("Select your portfolio risk level:", 1, 100, 5)
risk_levels = {
    "Very Low Risk": 73,
    "Low Risk": 54,
    "Medium Risk": 36,
    "High Risk": 18,
    "Very High Risk": 1
}
selected_risk_level = st.sidebar.selectbox(
    "Select your portfolio risk level:",
    options=list(risk_levels.keys())
)

# Retrieve the integer risk aversion value corresponding to the selected risk level
risk_aversion = risk_levels[selected_risk_level]

investment_amount = st.sidebar.number_input("Enter the amount you want to invest:", min_value=1000, step=500)


# Convert annualized standard deviation to covariance matrix
covariance_matrix = np.diag(df['Annualized_Std']**2)
extended_cov_matrix = np.pad(covariance_matrix, ((0, 1), (0, 1)), mode='constant', constant_values=0)

# Extended returns array including the risk-free asset
extended_returns = np.append(df['Total Expected Return (%)'].values, risk_free_return)

# Initial guess and bounds for the optimization
initial_guess = np.full(len(extended_returns), 1/len(extended_returns))
bounds = tuple((0, 1) for _ in range(len(extended_returns)))

# Constraints to ensure the sum of weights is 1
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

# Portfolio performance calculation function
def portfolio_performance(weights, returns, covariance_matrix):
    port_return = np.sum(weights * returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    return port_return, port_volatility

# Minimize utility function adjusted for risk aversion
def minimize_function(weights, risk_aversion, returns, covariance_matrix):
    port_return, port_volatility = portfolio_performance(weights, returns, covariance_matrix)
    utility = port_return - (risk_aversion / 2) * (port_volatility**2)
    return -utility

# Optimization process using user-defined risk aversion
result = minimize(minimize_function, initial_guess, args=(risk_aversion, extended_returns, extended_cov_matrix),
                  method='SLSQP', bounds=bounds, constraints=constraints)

if result.success:
    port_return, port_volatility = portfolio_performance(result.x, extended_returns, extended_cov_matrix)
    st.write(f"Optimized Portfolio for Risk Aversion {risk_aversion}:")
    st.write(f"Expected Return: {port_return:.2%}, Volatility: {port_volatility:.2%}")
    st.write("Portfolio Weights:")
    for i, ticker in enumerate(df['Ticker'].tolist() + ['Risk-Free Asset']):
        st.write(f"{ticker}: {result.x[i]*100:.2f}%")
else:
    st.error("Optimization did not converge")

# Optionally, you can include a plot here as well, or any additional analysis or data visualizations.

################################################################
#Graph code 
risk_level_results = []
for level, ra in risk_levels.items():
    result = minimize(minimize_function, initial_guess, args=(ra, extended_returns, extended_cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        port_return, port_volatility = portfolio_performance(result.x, extended_returns, extended_cov_matrix)
        risk_level_results.append({
            "Risk Level": level,
            "Risk Aversion": ra,
            "Expected Return": port_return,
            "Volatility": port_volatility,
            "Weights": result.x
        })

# Create the Plotly scatter plot
fig = go.Figure()

# Add scatter points for all risk aversion levels
for res in risk_level_results:
    fig.add_trace(go.Scatter(
        x=[res["Volatility"]],
        y=[res["Expected Return"]],
        text=[f"RA: {res['Risk Aversion']}<br>Return: {res['Expected Return']:.2%}<br>Volatility: {res['Volatility']:.2%}"],
        mode="markers",
        name=res["Risk Level"],
        marker=dict(
            size=12 if res["Risk Level"] == selected_risk_level else 8,
            symbol="star" if res["Risk Level"] == selected_risk_level else "circle"
        )
    ))

# Update layout to improve clarity
fig.update_layout(
    title="Risk Aversion Levels: Expected Return vs. Volatility",
    xaxis_title="Volatility (Standard Deviation)",
    yaxis_title="Expected Return",
    legend_title="Risk Levels",
    hovermode="closest"
)

# Display the plot in Streamlit
st.plotly_chart(fig, use_container_width=True)

################################################################
