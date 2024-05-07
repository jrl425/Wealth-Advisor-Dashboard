import streamlit as st 
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.optimize import minimize 

st.markdown("<p style='font-size:xx-Large; color:black;'>Report</p>", unsafe_allow_html=True)

import streamlit as st

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
This Streamlit application serves as a comprehensive financial planning tool, designed to assist users in making informed investment decisions and preparing for retirement. It leverages various Python libraries, including Streamlit for the user interface, Pandas for data manipulation, NumPy for numerical computations, Plotly for data visualization, and SciPy for optimization. The application begins by elucidating different risk tolerance levels and their implications on portfolio composition. It then employs historical market data to compute the optimal allocation of investments based on user-provided parameters such as initial investment, age, retirement age, and annual contribution. Utilizing SciPy's optimization capabilities, the application determines an investment strategy that maximizes returns while managing risk effectively. Subsequently, it conducts simulations to forecast portfolio growth over time, providing users with insights into the potential outcomes of their investment decisions. Furthermore, the application extends its functionality to retirement planning, allowing users to input parameters such as expected Social Security payments, life expectancy, and desired annual withdrawal rates. Through thorough analysis and simulation, users gain valuable insights to make prudent financial decisions aligned with their long-term goals.

# Data Overview:
The data was retrieved from Yahoo Finance using its built-in Python function. This data was then utilized to compute a 10-year average, which served as our expected return projections. Similarly, volatility was calculated based on this data. Additionally, we incorporated the annual dividend yield for each security into our analysis.
"""

# Render the Markdown content
st.markdown(markdown_text)

