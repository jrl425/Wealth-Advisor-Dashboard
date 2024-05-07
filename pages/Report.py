import streamlit as st 
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.optimize import minimize 

st.title("Report")


# Markdown content
markdown_text = """
# Overall Summary:

Welcome to Don Bowen Advisor’s Online Wealth Management Dashboard! Initiate your investment journey by completing the input survey located on the left side of your screen. Our algorithm will analyze your inputs to craft an investment portfolio tailored precisely to your goals. Visualize your portfolio allocation and explore the distribution of securities within your portfolio. Additionally, witness portfolio growth simulations spanning from your current age to your expected retirement, showcasing the evolution of your investments over time.

# Portfolio Allocation Summary:

## Risk Tolerance:

Our portfolio allocation service is guided by your risk preference, determined by the initial question in our input list. This selection shapes your resulting portfolio's risk level, ranging from very low to very high. A very low-risk tolerance yields a portfolio primarily comprised of debt (70%) and equity (30%), while a very high-risk tolerance allocates entirely to equity, with no debt.

## Graph:

Displayed in this section is a graph illustrating the expected return (y) and volatility (x) of portfolios across various risk levels. Analyze this graphical representation to inform your risk preference, with each plotted point corresponding to a specific risk level. Understand that riskier portfolios offer higher potential returns alongside increased volatility, while our service furnishes precise figures delineating these variations across risk levels.

## Portfolio Weights:

Beneath the graph, discover detailed information regarding the expected returns, volatility, and allocation of securities within your chosen portfolio. Our portfolio service integrates ten renowned market indexes, including SPY, VTI, IVV, QQQ, VXUS, IJH, IJR, VEA, VWO, and DIA. Extensive historical return analysis informs our predictions on the expected returns of these securities.

# Portfolio Simulation Summary:

## Inputs:

The Portfolio Simulation graph and figures hinge on numerous inputs gleaned from our input list. Initial investment amount, starting and retirement ages, and the desired number of simulations form the core inputs. Users may specify annual contribution amounts, with the option to adjust for yearly contribution growth rates ranging from 1% to 6%. A default growth rate of 2.2% is set to counteract projected inflation, aligning with the 30-year break-even inflation rate of 2.27%.

## Result:

Upon completing the Portfolio Simulation inputs, a graph showcasing the specified number of simulations illustrates the projected growth of your preferred portfolio from initiation to retirement age. Below this graph, numerical results from the simulations provide insights into the average, minimum, and maximum portfolio values at retirement age, offering a comprehensive understanding of your portfolio's potential trajectory.

# Retirement Plan Summary:

At Don Bowen Advisors, we recognize that portfolio management extends beyond retirement. Hence, our platform offers a retirement planning feature where users can input their estimated social security payment, life expectancy, and anticipated deduction percentage. A graph depicting portfolio balance throughout retirement is presented, alongside the safe annual withdrawal amount, empowering users to navigate their retirement journey with confidence.

# Code Overview:
This Streamlit application is a powerful financial planning tool, built using Python libraries like Pandas, NumPy, Plotly, and SciPy. It begins by guiding users through different risk tolerance levels and their impact on investment strategies. By analyzing historical market data, it computes the optimal investment allocation based on user inputs such as initial investment, age, retirement age, and annual contribution. Utilizing SciPy's optimization features, the app determines the investment strategy that maximizes returns while managing risk effectively. It then conducts simulations to forecast portfolio growth over time, providing users with valuable insights into potential investment outcomes. Additionally, the app extends its functionality to retirement planning, enabling users to input parameters like expected Social Security payments, life expectancy, and desired annual withdrawal rates. Through thorough analysis and simulation, users can make informed financial decisions aligned with their long-term goals.

# Data Overview:
The data was retrieved from Yahoo Finance using its built-in Python function. This data was then utilized to compute a 10-year average, which served as our expected return projections. Similarly, volatility was calculated based on this data. Additionally, we incorporated the annual dividend yield for each security into our analysis.
"""

# Render the Markdown content
st.markdown(markdown_text)

with st.expander("🥚"):
    st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdnBlc3BwcHZ1dXZ2eGhzYXlqcGV6MzA3c3pzcmxjazl0OHJuaGYwcCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/LdOyjZ7io5Msw/giphy.gif", caption="You when you use Don Bowen Advisors Retirement Dashboard", use_column_width=True)



