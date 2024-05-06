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
with st.expander("Click to show more"):
    st.write("""
    This is the additional text that will be displayed when the expander is clicked.
    
    You can add as much text or content here as you want.
    """)


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
    "Very Low": 73,
    "Low": 54,
    "Medium": 36,
    "High": 18,
    "Very High": 1
}
selected_risk_level = st.sidebar.selectbox(
    "Select Your Portfolio Risk Tolerance:",
    options=list(risk_levels.keys())
)
# Retrieve the integer risk aversion value corresponding to the selected risk level
risk_aversion = risk_levels[selected_risk_level]


st.sidebar.subheader("Portfolio Simulation Inputs")
investment_amount = st.sidebar.number_input("Enter the amount you want to invest:", min_value=1000, step=500)
age = st.sidebar.number_input("Age: ", min_value=18, step=1)
retirement_age = st.sidebar.number_input("Retirement Age (Must be greater than age): ", min_value=28, step=1)
simulations = st.sidebar.number_input("Number of Simulations", min_value=10, max_value=50, step=1)
annual_contribution = st.sidebar.number_input("Amount You Contribute Annually:", min_value=0, step=250)
percentage = st.sidebar.number_input("Annual Growth Rate (%):", min_value=0.0, max_value=6.0, value=2.2, step=0.1) / 100
st.sidebar.subheader("Retirement Plan")






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
## Commentted code
result = minimize(minimize_function, initial_guess, args=(risk_aversion, extended_returns, extended_cov_matrix),
                  method='SLSQP', bounds=bounds, constraints=constraints)

################################################################

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


###############################################################

#Everything below is portfolio simluation code
################################################################
# 
st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)
st.markdown("<p style='font-size:xx-Large; color:black;'>Portfolio Simulation</p>", unsafe_allow_html=True)
with st.expander("Click to show more"):
    st.write("""
    This is the additional text that will be displayed when the expander is clicked.
    
    You can add as much text or content here as you want.
    """)


if result.success:
    # Generate 25 simulations
    num_years = retirement_age - age
    simulation_results = np.zeros((simulations, num_years))
    
    # Assume 'percentage' is already defined, if not, define it or set a default value here.


    for i in range(simulations):
        annual_returns = np.random.normal(port_return, port_volatility, num_years)
        portfolio_values = [investment_amount * (1 + annual_returns[0])]
        
        for j in range(1, num_years):
            # Update portfolio value for the next year and add annual contribution
            next_value = (portfolio_values[-1] + annual_contribution) * (1 + annual_returns[j] + percentage)
            portfolio_values.append(next_value)
        
        simulation_results[i] = portfolio_values

    # Create a Plotly graph for the simulations
    simulation_fig = go.Figure()

    for i in range(simulations):
        hover_texts = [f"Age: {age+j} | Return: ${round(val, -2):,.0f}" for j, val in enumerate(simulation_results[i])]
        simulation_fig.add_trace(go.Scatter(
            x=list(range(age, retirement_age)),
            y=simulation_results[i],
            mode='lines',
            name=f'Simulation {i+1}',
            text=hover_texts,
            hoverinfo='text+x'
        ))

    simulation_fig.update_layout(
        title=f"Portfolio Growth Simulations from Age {age} to {retirement_age}",
        xaxis_title="Age",
        yaxis_title="Portfolio Value",
        legend_title="Simulations",
        hovermode="closest"
    )

    st.plotly_chart(simulation_fig, use_container_width=True)

    # Collect final portfolio values at retirement age
    final_values = [sim[-1] for sim in simulation_results]

    # Calculate and display aggregate statistics
    average_final_value = np.mean(final_values)
    min_final_value = np.min(final_values)
    max_final_value = np.max(final_values)

    st.write(f"Average Portfolio Value at Age {retirement_age}: ${average_final_value:,.2f}")
    st.write(f"Minimum Portfolio Value at Age {retirement_age}: ${min_final_value:,.2f}")
    st.write(f"Maximum Portfolio Value at Age {retirement_age}: ${max_final_value:,.2f}")

else:
    st.error("Failed to simulate portfolios. Optimization did not converge.")
st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)
################################################################


#################################################################
#

# User Inputs for Post-Retirement Planning
social_security_payment = st.sidebar.number_input("Estimated Annual Social Security Payment:", min_value=0, step=250)
expected_lifetime = st.sidebar.number_input("Expected Age to Live Until:", min_value=retirement_age, step=1, value = 85)

if 'average_final_value' in locals() and average_final_value > 0:
    # Calculate the number of years from retirement to expected death
    post_retirement_years = expected_lifetime - retirement_age

    # Simulation parameters
    simulation_results = np.zeros((simulations, post_retirement_years))

    # Estimate an initial sustainable annual deduction
    initial_annual_deduction = average_final_value / (expected_lifetime - retirement_age)

    # Simulate post-retirement scenarios
    for i in range(simulations):
        portfolio_values = [average_final_value]
        for j in range(1, post_retirement_years):
            # Calculate next year's balance considering returns, social security, and deductions
            growth = portfolio_values[-1] * (1 + percentage + np.random.normal(port_return, port_volatility))
            next_value = growth + social_security_payment - initial_annual_deduction
            portfolio_values.append(max(0, next_value))  # Ensure balance doesn't go negative

        simulation_results[i] = portfolio_values

    # Create a Plotly graph for the post-retirement simulations
    retirement_fig = go.Figure()

    for i in range(simulations):
        retirement_fig.add_trace(go.Scatter(
            x=list(range(retirement_age, expected_lifetime)),
            y=simulation_results[i],
            mode='lines',
            name=f'Simulation {i+1}',
            hoverinfo='text',
            text=[f"Age: {retirement_age + j} | Balance: ${round(val, -2):,.0f}" for j, val in enumerate(simulation_results[i])]
        ))

    retirement_fig.update_layout(
        title="Portfolio Balance During Retirement with Annual Withdrawals",
        xaxis_title="Age",
        yaxis_title="Portfolio Balance ($)",
        legend_title="Simulations",
        hovermode="closest"
    )

    st.plotly_chart(retirement_fig, use_container_width=True)

    # Calculate and display the mean withdrawal amount that keeps the portfolio balance positive
    final_balances = [sim[-1] for sim in simulation_results]
    if all(balance > 0 for balance in final_balances):  # Check if all simulations end with a positive balance
        st.success(f"Based on simulations, you can safely withdraw up to ${initial_annual_deduction:,.2f} annually.")
    else:
        st.error("Reduction in withdrawal amount required to avoid portfolio depletion.")

else:
    st.error("Average portfolio balance data is not available or insufficient to calculate retirement withdrawals.")

#################################################################
