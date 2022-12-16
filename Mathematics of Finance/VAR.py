import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
import numpy as np
import datetime as dt
from scipy.stats import norm
import matplotlib.pyplot as plt

#Tech sector, Real estate select sector, Retail, Gold
tickers = ['XLK', 'XLRE', 'XRT', 'GLD']

#Equal weighted portfolio
weights = np.array([.25, .25, .25, .25])

# Set an initial investment level = $1 Million
initial_investment = 1000000

# Download closing prices FOR TWO YEARS TOTAL BEFORE MARKET CRASH
data = pdr.get_data_yahoo(tickers, start="2018-02-20", end="2020-02-20")['Close']

# From the closing prices, calculate periodic returns
returns = data.pct_change()

print(returns.tail())

# Generate Var-Cov matrix
cov_matrix = returns.cov()
print(cov_matrix)

# Calculate mean returns for each stock
avg_rets = returns.mean()

# Calculate mean returns for portfolio overall,
# using dot product to
# normalize individual means against investment weights
port_mean = avg_rets.dot(weights)

# Calculate portfolio standard deviation
port_stdev = np.sqrt(weights.T.dot(cov_matrix).dot(weights))

# Calculate mean of investment
mean_investment = (1 + port_mean) * initial_investment

# Calculate standard deviation of investment
stdev_investment = initial_investment * port_stdev

# Choosing usual 95% confidence interval
conf_level1 = 0.05

# Using SciPy ppf method to generate values for the
# inverse cumulative distribution function to a normal distribution
# Plugging in the mean, standard deviation of our portfolio
# as calculated above
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
cutoff1 = norm.ppf(conf_level1, mean_investment, stdev_investment)

#Finally, we can calculate the VaR at our confidence interval
var_1d1 = initial_investment - cutoff1
print(var_1d1)

# DONT FORGET TO IMPORT import matplotlib.pyplot as plt
# Calculate n Day VaR
var_array = []
num_days = int(15)
for x in range(1, num_days+1):
    var_array.append(np.round(var_1d1 * np.sqrt(x),2))
    print(str(x) + " day VaR @ 95% confidence: " + str(np.round(var_1d1 * np.sqrt(x),2)))

# Build plot
plt.xlabel("Day #")
plt.ylabel("Max portfolio loss (USD)")
plt.title("Max portfolio loss (VaR) over 15-day period")
plt.plot(var_array, "r")

plt.show()