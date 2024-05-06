import streamlit as st

# Set custom page title
st.set_page_config(page_title="The Don Bowen Advisors Retirement Planning Dashboard")

# Display the image from the images folder
st.image("images/logo.png", use_column_width=True)

# Main app (Home page)
st.title("Retirement Planning Dashboard")
st.markdown("Welcome to the Don Bowen Advisors Dashboard! Use the navigation bar on the left to switch between different pages.")
st.markdown("<p style='font-size:medium; color:red;'>Disclaimer: The content provided on this site is for educational purposes only and should not be considered as financial advice; users are encouraged to consult with a qualified financial advisor before making any investment decisions.</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)
markdown_text_app = """ 
# Welcome
Welcome to the Don Bowen Advisor’s Online Wealth Management Dashboard! This platform offers a straightforward approach to investment management, allowing users to input their financial preferences and goals via a simple sidebar interface. Our algorithm then generates a customized investment portfolio designed to match individual risk tolerances and objectives. Visualize your portfolio allocation and explore the securities comprising it. Gain insights into portfolio growth through simulations spanning from your current age to retirement. With a focus on user-friendly design and actionable insights, our dashboard provides a practical tool for investors of all levels to make informed decisions and optimize their financial strategies. Take charge of your financial future today with the Don Bowen Advisor’s Online Wealth Management Dashboard.
# Our Motivation
Our team, comprised of Jimmy, Ben, and Danny, brings a wealth of experience and enthusiasm to this project. With backgrounds in finance and internships in wealth management, we're deeply passionate about empowering individuals to achieve their financial goals. As active investors ourselves, we understand the importance of effective retirement planning. Through this interactive dashboard, we aim to provide users with a comprehensive tool to plan for their retirement confidently and see their financial aspirations become reality.
# Our Team
"""
st.markdown(markdown_text_app)


st.image("images/three.png", caption = "Ben Ciancio (Finance) | Danny Pressler (Finance & BIS) | Jimmy Littley (Finance & BUAN )", use_column_width=True)











################################################################


# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objs as go
# from scipy.optimize import minimize 

# # Dashboard Title
# st.title("The Don Bowen Advisors Retirement Planning Dashboard")
# st.markdown("<p style='font-size:medium; color:red;'>Disclaimer: The content provided on this site is for educational purposes only and should not be considered as financial advice; users are encouraged to consult with a qualified financial advisor before making any investment decisions.</p>", unsafe_allow_html=True)

# # Dashboard expanders for additional information
# with st.expander("Click to show more"):
#     st.write("""
#     This is the additional text that will be displayed when the expander is clicked.
#     You can add as much text or content here as you want.
#     """)

# st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)
# st.markdown("<p style='font-size:xx-Large; color:black;'>Portfolio Allocation</p>", unsafe_allow_html=True)
# with st.expander("Click to Learn More About Your Risk Tolerance"):
#     st.write("""
# Very Low: A conservative portfolio with a focus on capital preservation, consisting of 70% risk-free assets and 30% equity.

# Low: A moderately conservative portfolio aimed at steady growth, maintaining a balance of risk and return.

# Medium: A balanced portfolio with a mix of risk-free assets and equity, suited for investors seeking moderate growth potential with manageable risk consisting of 40% risk-free assets and 60% equity. This is based on the simple investment strategy that allocates 60 percent of your holdings to stocks and 40 percent to bonds. 

# High: A growth-oriented portfolio with a significant allocation to equities, suitable for investors comfortable with higher levels of risk in pursuit of potentially higher returns.

# Very High: An aggressive portfolio entirely invested in equities, tailored for investors with a high risk tolerance seeking maximum growth potential over the long term.

#     """)

# # Load the ticker returns data
# df = pd.read_csv('inputs/index_data.csv')

# # 10-year treasury details
# risk_free_return = 0.04497
# risk_free_volatility = 0.0

# # Sidebar inputs
# st.sidebar.header("User Inputs for Wealth Management Dashboard")
# st.sidebar.subheader("Portfolio Allocation Inputs")
# risk_levels = {
#     "Very Low": 73,
#     "Low": 54,
#     "Medium": 36,
#     "High": 18,
#     "Very High": 1
# }
# selected_risk_level = st.sidebar.selectbox(
#     "Select Your Portfolio Risk Tolerance:",
#     options=list(risk_levels.keys())
# )
# risk_aversion = risk_levels[selected_risk_level]

# st.sidebar.subheader("Portfolio Simulation Inputs")
# investment_amount = st.sidebar.number_input("Initial Investment Amount:", min_value=1000, step=500)
# age = st.sidebar.number_input("Age: ", min_value=18, step=1)
# retirement_age = st.sidebar.number_input("Retirement Age (Must be greater than age): ", min_value=age+10, step=1)
# simulations = st.sidebar.number_input("Number of Simulations", min_value=10, max_value=50, step=1)
# annual_contribution = st.sidebar.number_input("Amount You Contribute Annually:", min_value=0, step=250)
# percentage = st.sidebar.number_input("Annual Growth Rate (%):", min_value=0.0, max_value=6.0, value=2.2, step=0.1) / 100

# # Covariance matrix and returns
# covariance_matrix = np.diag(df['Annualized_Std']**2)
# extended_cov_matrix = np.pad(covariance_matrix, ((0, 1), (0, 1)), mode='constant', constant_values=0)
# extended_returns = np.append(df['Total Expected Return (%)'].values, risk_free_return)

# # Optimization setup
# initial_guess = np.full(len(extended_returns), 1/len(extended_returns))
# bounds = tuple((0, 1) for _ in range(len(extended_returns)))
# constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

# def portfolio_performance(weights, returns, covariance_matrix):
#     port_return = np.sum(weights * returns)
#     port_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
#     return port_return, port_volatility

# def minimize_function(weights, risk_aversion, returns, covariance_matrix):
#     port_return, port_volatility = portfolio_performance(weights, returns, covariance_matrix)
#     utility = port_return - (risk_aversion / 2) * (port_volatility**2)
#     return -utility

# # Optimization process
# result = minimize(minimize_function, initial_guess, args=(risk_aversion, extended_returns, extended_cov_matrix),
#                   method='SLSQP', bounds=bounds, constraints=constraints)

# if result.success:
#     port_return, port_volatility = portfolio_performance(result.x, extended_returns, extended_cov_matrix)
#     st.write(f"Optimized Portfolio for {selected_risk_level} Risk Tolerance:")
#     st.write(f"Expected Return: {port_return:.2%}, Volatility: {port_volatility:.2%}")

#     # Filtering small weights and creating a pie chart
#     labels = df['Ticker'].tolist() + ['Risk-Free Asset']
#     weights = result.x * 100
#     filtered_labels = [label for label, weight in zip(labels, weights) if weight > 0.01]
#     filtered_weights = [weight for weight in weights if weight > 0.01]

#     # Plotting the pie chart
#     fig = go.Figure(data=[go.Pie(labels=filtered_labels, values=filtered_weights, hole=.3)])
#     fig.update_layout(title_text='Portfolio Allocation')
#     st.plotly_chart(fig, use_container_width=True)

#     # Expander for detailed weights
#     with st.expander("Click to show more"):
#         st.write("Detailed Portfolio Weights:")
#         for i, ticker in enumerate(labels):
#             if weights[i] > 0.01:  # Display only significant weights
#                 st.write(f"{ticker}: {weights[i]:.2f}%")
# else:
#     st.error("Optimization did not converge")
    
# #
# ###############################################################

# #Everything below is portfolio simluation code

# ################################################################
# # 

# st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)
# st.markdown("<p style='font-size:xx-Large; color:black;'>Portfolio Simulation</p>", unsafe_allow_html=True)
# with st.expander("Click to show more"):
#     st.write("""
#     This is the additional text that will be displayed when the expander is clicked.
    
#     You can add as much text or content here as you want.
#     """)


# if result.success:
#     # Generate 25 simulations
#     num_years = retirement_age - age
#     simulation_results = np.zeros((simulations, num_years))
    
#     # Assume 'percentage' is already defined, if not, define it or set a default value here.


#     for i in range(simulations):
#         annual_returns = np.random.normal(port_return, port_volatility, num_years)
#         portfolio_values = [investment_amount * (1 + annual_returns[0])]
        
#         for j in range(1, num_years):
#             # Update portfolio value for the next year and add annual contribution
#             next_value = (portfolio_values[-1] + annual_contribution) * (1 + annual_returns[j] + percentage)
#             portfolio_values.append(next_value)
        
#         simulation_results[i] = portfolio_values

#     # Create a Plotly graph for the simulations
#     simulation_fig = go.Figure()

#     for i in range(simulations):
#         hover_texts = [f"Age: {age+j} | Return: ${round(val, -2):,.0f}" for j, val in enumerate(simulation_results[i])]
#         simulation_fig.add_trace(go.Scatter(
#             x=list(range(age, retirement_age)),
#             y=simulation_results[i],
#             mode='lines',
#             name=f'Simulation {i+1}',
#             text=hover_texts,
#             hoverinfo='text+x'
#         ))

#     simulation_fig.update_layout(
#         title=f"Portfolio Growth Simulations from Age {age} to {retirement_age}",
#         xaxis_title="Age",
#         yaxis_title="Portfolio Value",
#         legend_title="Simulations",
#         hovermode="closest"
#     )

#     st.plotly_chart(simulation_fig, use_container_width=True)

#     # Collect final portfolio values at retirement age
#     final_values = [sim[-1] for sim in simulation_results]

#     # Calculate and display aggregate statistics
#     average_final_value = np.mean(final_values)
#     min_final_value = np.min(final_values)
#     max_final_value = np.max(final_values)

#     st.write(f"Average Portfolio Value at Age {retirement_age}: ${average_final_value:,.2f}")
#     st.write(f"Minimum Portfolio Value at Age {retirement_age}: ${min_final_value:,.2f}")
#     st.write(f"Maximum Portfolio Value at Age {retirement_age}: ${max_final_value:,.2f}")

# else:
#     st.error("Failed to simulate portfolios. Optimization did not converge.")
# st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)

# #
# ################################################################

# #Everything below is retirement simluation code

# #################################################################
# #

# # Streamlit setup for user input
# st.markdown("<p style='font-size:xx-Large; color:black;'>Retirement Simulation</p>", unsafe_allow_html=True) 
# with st.expander("Click to show more"):
#     st.write("""
#     This is the additional text that will be displayed when the expander is clicked.
    
#     You can add as much text or content here as you want.
#     """)

# # User Inputs for Post-Retirement Planning
# social_security_payment = st.sidebar.number_input("Estimated Annual Social Security Payment:", min_value=0, step=250)
# expected_lifetime = st.sidebar.number_input("Expected Age to Live Until:", min_value=retirement_age, step=1, value=85)
# deduction_pct = (st.sidebar.number_input("Anticipated Annual Withdrawal Rate:", min_value=0.0, step=.10, value=2.0))/100


# if 'average_final_value' in locals() and average_final_value > 0:
#     # Calculate the number of years from retirement to expected death
#     post_retirement_years = expected_lifetime - retirement_age

#     # Simulation parameters
#     simulation_results = np.zeros((simulations, post_retirement_years))

#     # Estimate an initial sustainable annual deduction based on a fixed withdrawal rate
#     initial_annual_deduction = average_final_value * deduction_pct  # Example: 4% withdrawal rate

#     # Simulate post-retirement scenarios
#     for i in range(simulations):
#         portfolio_values = [average_final_value]
#         for j in range(1, post_retirement_years):
#             # Calculate next year's balance considering returns and deductions
#             expected_growth = np.random.normal(port_return/100, port_volatility/100)
#             growth = portfolio_values[-1] * (1 + expected_growth)
#             next_value = growth - initial_annual_deduction + social_security_payment
#             portfolio_values.append(max(0, next_value))  # Ensure balance doesn't go negative




#         simulation_results[i] = portfolio_values

#     # Create a Plotly graph for the post-retirement simulations
#     retirement_fig = go.Figure()

#     for i in range(simulations):
#         retirement_fig.add_trace(go.Scatter(
#             x=list(range(retirement_age, expected_lifetime)),
#             y=simulation_results[i],
#             mode='lines',
#             name=f'Simulation {i+1}',
#             hoverinfo='text',
#             text=[f"Age: {retirement_age + j} | Balance: ${round(val, -2):,.0f}" for j, val in enumerate(simulation_results[i])]
#         ))

#     retirement_fig.update_layout(
#         title="Portfolio Balance During Retirement with Annual Withdrawals",
#         xaxis_title="Age",
#         yaxis_title="Portfolio Balance ($)",
#         legend_title="Simulations",
#         hovermode="closest"
#     )

#     st.plotly_chart(retirement_fig, use_container_width=True)

#     # Calculate and display the mean withdrawal amount that keeps the portfolio balance positive
#     final_balances = [sim[-1] for sim in simulation_results]
#     if all(balance > 0 for balance in final_balances):  # Check if all simulations end with a positive balance
#         st.success(f"Based on simulations, you can safely withdraw up to ${initial_annual_deduction:,.2f} annually.")
#     else:
#         st.error("Reduction in withdrawal amount required to avoid portfolio depletion.")

# else:
#     st.error("Average portfolio balance data is not available or insufficient to calculate retirement withdrawals.")

# ##################################################################
