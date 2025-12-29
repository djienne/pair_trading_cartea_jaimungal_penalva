### 1. Import Dependencies and Get Data

import numpy as np
import pandas as pd
try:
    import pandas_datareader.data as web
except ImportError:
    web = None
try:
    import yfinance as yf
except ImportError:
    yf = None
import datetime
import scipy.stats as sc
import scipy.optimize as sco
import os
import matplotlib.pyplot as plt

# Define function to fetch data
def get_data(stocks, start, end):
    # Note: Yahoo Finance API often changes. If pandas_datareader fails, 
    # yfinance can be used as an alternative override.
    if web is not None:
        try:
            return web.DataReader(stocks, 'yahoo', start, end)
        except Exception:
            pass
    if yf is not None:
        return yf.download(stocks, start=start, end=end)
    raise ImportError(
        "Missing data source. Install pandas_datareader or yfinance."
    )

# Set time range
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=7000) # Roughly 2003

# Fetch S&P 500 data
stock_prices = get_data('^GSPC', start_date, end_date)

# Display head
print(stock_prices.head())

### 2. Calculate Log Returns and Realized Volatility

# Calculate Log Returns
log_returns = np.log(stock_prices.Close / stock_prices.Close.shift(1)).dropna()

# Calculate Rolling Volatility (Annualized)
TRADING_DAYS = 40
volatility = log_returns.rolling(window=TRADING_DAYS).std() * np.sqrt(252)
volatility = volatility.dropna()

# Plot the realized volatility
plt.figure(figsize=(10, 6))
plt.plot(volatility)
plt.title("S&P 500 Rolling Volatility")
if os.environ.get("SHOW_PLOTS", "0") == "1":
    plt.show()
else:
    plt.tight_layout()
    plt.savefig("rolling_volatility.png", dpi=150)
    plt.close()

### 3. Ornstein-Uhlenbeck Calibration (MLE)

#This section implements the exact solution derived in the video to find the optimal parameters ($\kappa, \theta, \sigma$) that fit the historical volatility data.

def mu(x, dt, kappa, theta):
    """
    Calculates the expected mean of the process at the next time step
    based on the exact solution of the SDE.
    """
    ekt = np.exp(-kappa * dt)
    return x * ekt + theta * (1 - ekt)

def std(dt, kappa, sigma):
    """
    Calculates the standard deviation of the process at the next time step
    based on the exact solution of the SDE.
    """
    e2kt = np.exp(-2 * kappa * dt)
    return sigma * np.sqrt((1 - e2kt) / (2 * kappa))

def log_likelihood_OU(theta_hat, x):
    """
    Calculates the negative log likelihood of the observed data given 
    parameters theta_hat.
    """
    kappa = theta_hat[0]
    theta = theta_hat[1]
    sigma = theta_hat[2]
    
    # Time step (daily data annualized)
    dt = 1 / 252
    
    # x_t is the current value, x_dt is the next value
    x_dt = x[1:]
    x_t = x[:-1]
    
    # Calculate Mean and Std Dev for the transition probability
    mu_OU = mu(x_t, dt, kappa, theta)
    sigma_OU = std(dt, kappa, sigma)
    
    # Calculate sum of log likelihoods
    # We sum the logs of the PDF of the observed next step given the current step
    # Use logpdf directly to avoid underflow/zero issues.
    l_theta = np.sum(sc.norm.logpdf(x_dt, loc=mu_OU, scale=sigma_OU))
    
    return -l_theta

def kappa_pos(theta_hat):
    return theta_hat[0]

def sigma_pos(theta_hat):
    return theta_hat[2]

# Prepare data for optimization
vol = np.array(volatility)

# Constraints: Kappa and Sigma must be positive
cons_set = [
    {'type': 'ineq', 'fun': kappa_pos},
    {'type': 'ineq', 'fun': sigma_pos}
]

# Initial guess [Kappa, Theta, Sigma]
theta0 = [1, 0.2, 0.2]

# Run Optimization
opt = sco.minimize(fun=log_likelihood_OU, 
                   x0=theta0, 
                   args=(vol,), 
                   constraints=cons_set, 
                   method='SLSQP')

kappa_hat = opt.x[0]
theta_hat = opt.x[1]
sigma_hat = opt.x[2]

print("Optimized Parameters:")
print(f"Kappa: {kappa_hat:.4f}")
print(f"Theta (Long term mean): {theta_hat:.4f}")
print(f"Sigma (Vol of Vol): {sigma_hat:.4f}")

### 4. Simulating the Process

#Using the calibrated parameters to simulate future volatility paths.

# Simulation parameters
years = 2
dt = 1/252
N = int(years / dt) # Number of time steps
M = 100 # Number of paths (simulations)

# Initialize array for simulations
# rows = paths, columns = time steps
vol_sim = np.full(shape=(M, N), fill_value=vol[-1])

# Run simulation loop
# We start from range 1 because index 0 is the initial fill_value (last observed vol)
for t in range(1, N):
    # Get previous values
    x_prev = vol_sim[:, t-1]
    
    # Calculate drift and diffusion components using exact solution logic
    drift = mu(x_prev, dt, kappa_hat, theta_hat)
    diffusion = std(dt, kappa_hat, sigma_hat)
    
    # Generate random shocks
    Z = np.random.normal(size=M)
    
    # Update state
    vol_sim[:, t] = drift + diffusion * Z

# Plot simulations
plt.figure(figsize=(10, 6))
plt.plot(vol_sim.T, alpha=0.4)
plt.title(f"Ornstein-Uhlenbeck Simulation ({years} Years)")
plt.ylabel("Volatility")
plt.xlabel("Time Steps (Days)")
if os.environ.get("SHOW_PLOTS", "0") == "1":
    plt.show()
else:
    plt.tight_layout()
    plt.savefig("ou_simulation.png", dpi=150)
    plt.close()
