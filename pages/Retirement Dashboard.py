import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.optimize import minimize 

# st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)
st.markdown("<p style='font-size:xx-Large; color:black;'>Portfolio Allocation</p>", unsafe_allow_html=True)
with st.expander("Click to Learn More About Your Risk Tolerance"):
    st.write("""
Very Low: A conservative portfolio with a focus on capital preservation, consisting of 70% risk-free assets and 30% equity.

Low: A moderately conservative portfolio aimed at steady growth, maintaining a balance of risk and return.

Medium: A balanced portfolio with a mix of risk-free assets and equity, suited for investors seeking moderate growth potential with manageable risk consisting of 40% risk-free assets and 60% equity. This is based on the simple investment strategy that allocates 60 percent of your holdings to stocks and 40 percent to bonds. 

High: A growth-oriented portfolio with a significant allocation to equities, suitable for investors comfortable with higher levels of risk in pursuit of potentially higher returns.

Very High: An aggressive portfolio entirely invested in equities, tailored for investors with a high risk tolerance seeking maximum growth potential over the long term.

    """)

# Load the ticker returns data
df = pd.read_csv('inputs/index_data.csv')

# 10-year treasury details
risk_free_return = 0.04497
risk_free_volatility = 0.0

# Sidebar inputs
st.sidebar.header("User Inputs for Wealth Management Dashboard")
st.sidebar.subheader("Portfolio Allocation Inputs")
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
risk_aversion = risk_levels[selected_risk_level]

st.sidebar.subheader("Portfolio Simulation Inputs")
investment_amount = st.sidebar.number_input("Initial Investment Amount:", min_value=1000, step=500)
age = st.sidebar.number_input("Age: ", min_value=18, step=1)
retirement_age = st.sidebar.number_input("Retirement Age (Must be greater than age): ", min_value=age+10, value=60, step=1)
simulations = st.sidebar.number_input("Number of Simulations", min_value=50, max_value=150,value=100, step=1)
annual_contribution = st.sidebar.number_input("Annual Contribution:", min_value=0, value=2500, step=250)
percentage = st.sidebar.number_input("Annual Contribution Growth Rate (%):", min_value=0.0, max_value=6.0, value=2.2, step=0.1) / 100
st.sidebar.subheader("Retirement Simulation Inputs")


# Covariance matrix and returns
covariance_matrix = np.diag(df['Annualized_Std']**2)
extended_cov_matrix = np.pad(covariance_matrix, ((0, 1), (0, 1)), mode='constant', constant_values=0)
extended_returns = np.append(df['Total Expected Return (%)'].values, risk_free_return)

# Optimization setup
initial_guess = np.full(len(extended_returns), 1/len(extended_returns))
bounds = tuple((0, 1) for _ in range(len(extended_returns)))
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

def portfolio_performance(weights, returns, covariance_matrix):
    port_return = np.sum(weights * returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    return port_return, port_volatility

def minimize_function(weights, risk_aversion, returns, covariance_matrix):
    port_return, port_volatility = portfolio_performance(weights, returns, covariance_matrix)
    utility = port_return - (risk_aversion / 2) * (port_volatility**2)
    return -utility

# Optimization process
result = minimize(minimize_function, initial_guess, args=(risk_aversion, extended_returns, extended_cov_matrix),
                  method='SLSQP', bounds=bounds, constraints=constraints)

if result.success:
    port_return, port_volatility = portfolio_performance(result.x, extended_returns, extended_cov_matrix)
    st.write(f"Optimized Portfolio for {selected_risk_level} Risk Tolerance:")
    st.write(f"Expected Return: {port_return:.2%}, Volatility: {port_volatility:.2%}")

    # Filtering small weights and creating a pie chart
    labels = df['Ticker'].tolist() + ['Risk-Free Asset']
    weights = result.x * 100
    filtered_labels = [label for label, weight in zip(labels, weights) if weight > 0.01]
    filtered_weights = [weight for weight in weights if weight > 0.01]

    # Plotting the pie chart
    fig = go.Figure(data=[go.Pie(labels=filtered_labels, values=filtered_weights, hole=.3)])
    fig.update_layout(title_text='Portfolio Allocation')
    st.plotly_chart(fig, use_container_width=True)

    # Expander for detailed weights
    with st.expander("Click to Show Detailed Portfolio Weights"):
        st.write("Detailed Portfolio Weights:")
        for i, ticker in enumerate(labels):
            if weights[i] > 0.01:  # Display only significant weights
                st.write(f"{ticker}: {weights[i]:.2f}%")
else:
    st.error("Optimization did not converge")
    
with st.expander("Click to Learn More About These Securities"):
    st.write("""
SPY: Tracks the performance of the S&P 500 Index, representing 500 of the largest publicly traded companies in the United States.

VTI: Provides exposure to the entire U.S. stock market, comprising large, mid, and small-cap stocks.

IVV: Offers investors exposure to the S&P 500 Index, consisting of large-cap U.S. stocks.

QQQ: Focuses on tracking the performance of the Nasdaq-100 Index, composed of 100 of the largest non-financial companies listed on the Nasdaq Stock Market.

VXUS: Diversifies investment across international developed and emerging markets outside of the United States.

IJH: Tracks the performance of mid-cap U.S. stocks, providing exposure to companies with market capitalizations between those of large and small-cap stocks.

IJR: Offers exposure to small-cap U.S. stocks, representing companies with smaller market capitalizations.

VEA: Invests in developed markets outside of the United States, excluding Canada and the U.S.

VWO: Provides exposure to emerging market equities, comprising stocks from countries with developing economies.

DIA: Tracks the performance of the Dow Jones Industrial Average, representing 30 large-cap U.S. stocks.

Risk Free Rate: The 10-year Treasury is a government bond issued by the United States Department of the Treasury with a fixed interest rate and a maturity of 10 years, commonly used as a benchmark for long-term interest rates and as a safe-haven investment.
""")
    
#
###############################################################

#Everything below is portfolio simluation code

################################################################
# 

st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)
st.markdown("<p style='font-size:xx-Large; color:black;'>Portfolio Simulation</p>", unsafe_allow_html=True)
with st.expander("Click to Learn More About Your Inputs"):
    st.write("""
Initial Investment Amount: This input allows you to specify the initial amount of money you want to invest.

Age: Enter your current age in years.

Retirement Age (Must be greater than age): Input your expected retirement age. It must be greater than your current age.

Number of Simulations: Choose the number of simulations to run for portfolio analysis. Due to volatility, each simulation will be different. More simulations mean the model accounts for greater uncertainty, which may help the model be more accurate.

Annual Contribution: Enter the amount of money you plan to contribute to your investment annually.

Annual Contribution Growth Rate (%): Set the annual contribution growth rate for your investment portfolio as a percentage. This value is preset to 2.20%, using the 30 year B.I.R. which essentially represents the average annual inflation rate over a 30-year period. 
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

#
################################################################

#Everything below is retirement simluation code

#################################################################
#

# Streamlit setup for user input
st.markdown("<p style='font-size:xx-Large; color:black;'>Retirement Simulation</p>", unsafe_allow_html=True) 
with st.expander("Click to Learn More About Your Inputs"):
    st.write("""
Estimated Annual Social Security Payment: Input your estimated annual Social Security payment amount.

Expected Age to Live Until: Enter the age you expect to live until. This helps in retirement planning.

Anticipated Annual Withdrawal Rate: Set the anticipated annual withdrawal rate for your retirement funds. Preset to follow the 4% rule.
    """)

# User Inputs for Post-Retirement Planning
social_security_payment = st.sidebar.number_input("Estimated Annual Social Security Payment:", min_value=0, value=15000, step=250)
expected_lifetime = st.sidebar.number_input("Expected Age to Live Until:", min_value=retirement_age, step=1, value=85)
deduction_pct = (st.sidebar.number_input("Anticipated Annual Withdrawal Rate:", min_value=0.0, step=.10, value=4.0))/100


if 'average_final_value' in locals() and average_final_value > 0:
    # Calculate the number of years from retirement to expected death
    post_retirement_years = expected_lifetime - retirement_age

    # Simulation parameters
    simulation_results = np.zeros((simulations, post_retirement_years))

    # Estimate an initial sustainable annual deduction based on a fixed withdrawal rate
    initial_annual_deduction = average_final_value * deduction_pct  # Example: 4% withdrawal rate

    # Simulate post-retirement scenarios
    for i in range(simulations):
        portfolio_values = [average_final_value]
        for j in range(1, post_retirement_years):
            # Calculate next year's balance considering returns and deductions
            expected_growth = np.random.normal(port_return/100, port_volatility/100)
            growth = portfolio_values[-1] * (1 + expected_growth)
            next_value = growth - initial_annual_deduction + social_security_payment
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

##################################################################
