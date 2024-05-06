import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import plotly.graph_objs as go

# Add a large title to the dashboard
st.title("The Don Bowen Advisors", anchor='center')
st.markdown("<p style='font-size:medium; color:red;'>Disclaimer: This is not financial advice.</p>", unsafe_allow_html=True)
st.markdown("<p style='font-size:Large; color:black;'>(Put summary here please)</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)
st.markdown("<p style='font-size:xx-Large; color:black;'>Portfolio Allocation</p>", unsafe_allow_html=True)

# Load the ticker returns data
df = pd.read_csv('inputs/index_data.csv')

#10 year treasury
risk_free_return = 0.04497
risk_free_volatility = 0.0

# Sidebar for user inputs
st.sidebar.header("User Inputs for Wealth Management Dashboard")
st.sidebar.subheader("Portfolio Allocation Inputs")
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


st.sidebar.subheader("Portfolio Simulation Inputs")
investment_amount = st.sidebar.number_input("Enter the amount you want to invest:", min_value=1000, step=500)
age = st.sidebar.number_input("Age: ", min_value=18, step=1)
retirement_age = st.sidebar.number_input("Retirement Age (Must be greater than age): ", min_value=28, step=1)





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

################################################################
# Prints out weights
if result.success:
    port_return, port_volatility = portfolio_performance(result.x, extended_returns, extended_cov_matrix)
    st.write(f"Optimized Portfolio for Risk Aversion {risk_aversion}:")
    st.write(f"Expected Return: {port_return:.2%}, Volatility: {port_volatility:.2%}")
    st.write("Portfolio Weights:")
    for i, ticker in enumerate(df['Ticker'].tolist() + ['Risk-Free Asset']):
        st.write(f"{ticker}: {result.x[i]*100:.2f}%")
else:
    st.error("Optimization did not converge")


################################################################

#Everything below is portfolio simluation code
################################################################
# 
st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)
st.markdown("<p style='font-size:xx-Large; color:black;'>Portfolio Simulation</p>", unsafe_allow_html=True)

if result.success:
    # Generate 5 simulations
    num_years = retirement_age - age
    simulations = 5
    simulation_results = np.zeros((simulations, num_years))

    for i in range(simulations):
        annual_returns = np.random.normal(port_return, port_volatility, num_years)
        portfolio_values = [investment_amount * (1 + annual_returns[0])]
        
        for j in range(1, num_years):
            portfolio_values.append(portfolio_values[-1] * (1 + annual_returns[j]))
        
        simulation_results[i] = portfolio_values

    # Create a Plotly graph for the simulations
    simulation_fig = go.Figure()

    for i in range(simulations):
        simulation_fig.add_trace(go.Scatter(
            x=list(range(age, retirement_age)),
            y=simulation_results[i],
            mode='lines',
            name=f'Simulation {i+1}'
        ))

    simulation_fig.update_layout(
        title=f"Portfolio Growth Simulations from Age {age} to {retirement_age}",
        xaxis_title="Age",
        yaxis_title="Portfolio Value",
        legend_title="Simulations",
        hovermode="closest"
    )

    st.plotly_chart(simulation_fig, use_container_width=True)

    # Display final portfolio values at retirement age
    st.write("Final Portfolio Values at Retirement:")
    final_values = []
    for i in range(simulations):
        final_value = simulation_results[i][-1]
        final_values.append(final_value)
        st.write(f"Simulation {i+1} Portfolio Value at Year {retirement_age}: ${final_value:,.2f}")

    # Calculate and display the average final portfolio value
    average_final_value = np.mean(final_values)
    st.write(f"Average Portfolio Value at Year {retirement_age}: ${average_final_value:,.2f}")

else:
    st.error("Failed to simulate portfolios. Optimization did not converge.")
################################################################
