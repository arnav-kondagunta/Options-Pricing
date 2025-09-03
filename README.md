# Options-Pricing
I built an Options Pricing Model that integrates a Monte Carlo simulation with my [Binomial Asset Pricing Model](https://github.com/arnav-kondagunta/Binomial-Model) and [Black-Scholes Model project](https://github.com/arnav-kondagunta/Black-Scholes). It uses data from Yahoo Finance to fetch stock data and the Federal Reserve Economic Data (FRED) to fetch risk-free treasury rates. The following describes how each model works.

**Black-Scholes Model:** Derives the price of a call/put option given a stock ticker using a formula created by Fischer Black, Myron Scholes, and Robert Merton. 

**Binomial Asset Pricing Model:** Builds a binomial tree model to map out the stock price paths and uses the option payoff function to create a second tree model of the corresponding option price paths. Then, the price of the option is given by the discounted conditional expectation of the option price at the initial time step.

**Monte Carlo Simulation:** Projects out a large number of random stock price pathways, calculates each path's options payout, and averages the discounted payoff to find the estimated optoin value.

## Files
Each Jupyter Noteboook contains each separate framework, and the Python file uses Streamlit to create an interactive web application. 
