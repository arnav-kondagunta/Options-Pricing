#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import yfinance as yf
import os
from fredapi import Fred
from dotenv import load_dotenv, dotenv_values
import datetime
import streamlit as st


#assigning env variables
load_dotenv()
FED_API_KEY = os.getenv('FED_API_KEY')
FED_RATES = ['DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30', 'DGS3MO', 'DGS6MO', 'DGS1MO', 'DGS2MO', 'DGS3MO', 'DGS6MO', 'T10YIE', 'FEDFUNDS']

#fetch risk free rate from fred data
def get_fred_data(rate_id):
    fred = Fred(api_key=FED_API_KEY)
    risk_free_rate = fred.get_series(rate_id).iloc[-1] / 100
    return risk_free_rate

#fetch yfinance stock data
def get_stock_data(ticker, period='1y'):
    stock = yf.Ticker(ticker)
    stock_price = stock.history(period='1d')['Close'][0]
    hist = stock.history(period=period)
    hist['returns'] = hist['Close'].pct_change()
    volatility = hist['returns'].std() * (252 ** 0.5)
    dividend = stock.info['dividendYield']
    return stock_price, volatility, hist, dividend

def get_time_period(expire_date):
    today = datetime.date.today()
    time_period = abs((expire_date - today).days)
    return time_period / 252

#black-scholes model
def black_scholes_calc(S0, K, r, T, sd, option_type='Call'):
    '''
    S0 = initial stock price
    K = strike price
    r = risk-free rate
    T = time until expiry
    sd = volatility
    '''

    #First, determine d1 and d2
    d1 = 1/(sd*np.sqrt(T)) * (np.log(S0/K) + (r+sd**2/2)*T)
    d2 = d1 - sd*np.sqrt(T)

    #Next, find norm cdf of d1 and d2
    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)

    n_d1 = norm.cdf(-d1)
    n_d2 = norm.cdf(-d2)

    #Then, find call and put value
    call = round(nd1*S0 - nd2*K*np.exp(-r*T), 2)
    put = round(K*np.exp(-r*T)*n_d2 - S0*n_d1, 2)

    if option_type=="Call":
        return call
    elif option_type=="Put":
        return put
    else:
        print("Wrong option type specified.")

def binomial_model(S0, K, r, T, n, sd, div=0, option_type="Call", exercise_type="European"):
    '''
    S0 = initial stock price
    K = strike price
    r = risk-free rate
    T = time until expiry
    n = number of time steps
    sd = volatility
    div = dividend yield
    option_type = call or put option
    exercise_type = european or american
    '''

    #Assign values to delta T, up factor, down factor, probabilty of going up (p), and probability of going down(q)
    deltaT = T / n
    up = np.exp(sd*np.sqrt(deltaT))
    down = 1 / up
    p = (np.exp((r-div)*deltaT) - down) / (up - down)

    #Create stock price tree at each time step
    stock_tree = np.zeros([n+1,n+1])
    for i in range(n+1):
        for j in range(i+1):
            stock_tree[i, j] = S0*(up**j)*(down**(i-j))
    
    #Create option value tree at each time step
    option_value = np.zeros([n+1,n+1])

    #Calculate value of option in the last time step based on each possible stock price in last time step
    if option_type == "Call":
        for j in range(n+1):
            option_value[n,j] = max(stock_tree[n,j] - K, 0)  # Call payoff is stock price minus strike price (or zero if not in the money)
    elif option_type == "Put":
        for j in range(n+1):
            option_value[n,j] = max(K - stock_tree[n,j], 0)  # Put payoff is strike price minus stock price (or zero if not in the money)
    else:
        raise ValueError("Wrong option type specified")
    
    #Calculate values of option in previous time steps using expected value formula
    for i in reversed(range(n)):
        for j in range(i+1):
            continuation = np.exp(-r*deltaT)*(p*option_value[i+1,j+1]+(1-p)*option_value[i+1,j])
            if exercise_type == "American":
                if option_type == "Call":
                    intrinsic = max(stock_tree[i,j] - K, 0)
                if option_type == "Put":
                    intrinsic = max(K - stock_tree[i,j],0)
                option_value[i,j] = max(continuation,intrinsic)
            elif exercise_type == "European":
                option_value[i,j] = continuation
            else:
                raise ValueError("Wrong exercise type specified.")
    return option_value, stock_tree 
        
def monte_carlo(S0, K, r, T, n, m, sd, option_type="Call"):
    '''
    S0 = initial stock price
    K = strike price
    r = risk-free rate
    T = time until expiry
    n = number of time steps
    m = number of price paths
    sd = volatility
    option_type = call or put option
    exercise_type = european or american
    '''

    #Discretize the SDE
    deltaT = T / n

    stock_tree = np.zeros((n+1, m))
    stock_tree[0] = S0
    for t in range(1,n+1):
        stock_tree[t] = stock_tree[t-1]*np.exp((r -0.5*sd**2)*deltaT+sd*np.sqrt(deltaT)*np.random.standard_normal(m))
    
    #Calculate payoff for each path and find discounted value of option
    payoffs = np.zeros(m)
    if option_type == "Call":
        for i in range(m):
            payoffs[i] = max(stock_tree[n,i] - K, 0)
    if option_type == "Put":
        for i in range(m):
            payoffs[i] = max(K - stock_tree[n,i], 0)
    option_price = np.exp(-r*T) * np.mean(payoffs)
    standard_error = np.exp(-r*T) * payoffs.std(ddof=1) / np.sqrt(m)

    return option_price,stock_tree,standard_error


#streamlit title
st.title("Options Pricing Models")

#data input
ticker = st.text_input("Enter the ticker for your stock:", key='ticker', value='AAPL')
strike_price = float(st.number_input("Enter the strike price of the option:", key='strike_price', value=180))
expire_date = st.date_input("Enter the expiration date of your option:", key='expire_date', value=datetime.date(2028, 1 ,1))
risk_free_rates = st.selectbox("Risk-free rate options", FED_RATES, key="rates", index=5)
num_steps = int(st.number_input("Enter the number of time steps of the model:", key='num_steps', value=100))
opt_type = st.selectbox("What is your option type?", ("Call", "Put"))
exercise_type = st.selectbox("What is the exercise type?", ("European", "American"))


if ticker:
    stock_price, volatility, hist, dividend = get_stock_data(st.session_state.ticker)
else:
    st.write('Please enter a valid stock ticker symbol.')

if expire_date:
    time = get_time_period(st.session_state.expire_date)
else:
    st.write('Please enter a valid date.')

if risk_free_rates:
    rate = get_fred_data(st.session_state.rates)

if not num_steps:
    st.write('Please enter a valid number of time steps.')

st.write("### Inputs")
st.write(f"Ticker: {ticker}")
st.line_chart(hist['Close'])
st.write(f"Spot price: ${stock_price:.2f}")
st.write(f"Strike price: ${strike_price:.2f}")
st.write(f"Time: {time:.2} years")
st.write(f"Risk-free rate ({st.session_state.rates}): {rate:.2%}")
st.write(f"Number of time steps: {num_steps}")
st.write(f"Volatility: {volatility:.2%}")

st.write("### Outputs")
if ticker and stock_price and rate and time and volatility and num_steps and exercise_type=="European":
    black_scholes_opt_price = black_scholes_calc(stock_price, strike_price, rate, time, volatility, option_type=opt_type)
    binomial_opt_tree, binomial_stock_tree = binomial_model(stock_price, strike_price, rate, time, num_steps, volatility, option_type=opt_type,exercise_type=exercise_type)
    monte_carlo_opt_price, monte_carlo_paths_tree, error = monte_carlo(stock_price, strike_price, rate, time, num_steps, 100000, volatility, option_type=opt_type)
    st.write(f"(Black-Scholes Model) The value of your option is ${round(black_scholes_opt_price,2)}.")
    st.write(f"(Binomial Model) The value of your option is ${round(binomial_opt_tree[0,0],2)}.")
    st.write(f"(Monte Carlo) The value of your option is ${round(monte_carlo_opt_price,2)}.")
elif ticker and stock_price and rate and time and volatility and num_steps and exercise_type=="American":
    binomial_opt_tree, binomial_stock_tree = binomial_model(stock_price, strike_price, rate, time, num_steps, volatility, option_type=opt_type,exercise_type=exercise_type)
    st.write(f"(Binomial Model) The value of your option is ${round(binomial_opt_tree[0,0],2)}.")
else:
    st.write('Error. Please try again.')
