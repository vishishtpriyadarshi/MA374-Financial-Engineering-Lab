"""
Kindly install these libraries before executing this code:
  1. numpy
  2. matplotlib
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import random

# if using a Jupyter notebook, kindly uncomment the following line:
# %matplotlib inline

def GBM_model(S_0, mu, sigma, n):
  dt = 1.0/252
  W_t = np.random.normal(0, 1, n)
  prices = []

  for i in range(n):
    S_t = S_0 * math.exp( (mu - sigma**2)*dt + sigma*math.sqrt(dt)*W_t[i])
    prices.append(S_t)
    S_0 = S_t

  return prices


def compute_path(S_0, mu, sigma, heading):
  n = 252
  x = np.arange(n)

  for i in range(10):
    prices = GBM_model(S_0, mu, sigma, n)
    plt.plot(x, prices)
  
  plt.xlabel('time, t (in days)')
  plt.ylabel('Stock prices, S(t)')
  plt.title(heading)
  plt.show()


def asian_option_price(S_0, r, sigma, K, max_iter = 1000, path_length = 126, n = 126):
  dt = 1/252
  call_option_payoff, put_option_payoff = [], []

  for i in range(max_iter):
    S = GBM_model(S_0, r, sigma, path_length)
    V_call = max(np.mean(S) - K, 0)
    V_put = max(K - np.mean(S), 0)

    call_option_payoff.append(math.exp(-r*n*dt) * V_call)
    put_option_payoff.append(math.exp(-r*n*dt) * V_put)

  return np.mean(call_option_payoff), np.mean(put_option_payoff), np.var(call_option_payoff), np.var(put_option_payoff)


def variation_with_S0(r, sigma, K, display=True):
  S0 = np.linspace(50, 150, num=250)
  call, put = [], []

  for i in S0:
    call_price, put_price, _, _ = asian_option_price(i, r, sigma, K, 500, 150, 100)
    call.append(call_price)
    put.append(put_price)
  
  if display == True:
    plt.plot(S0, call)
    plt.xlabel("Initial asset price (S0)")
    plt.ylabel("Asian call option price")
    plt.title("Dependence of Asian Call Option on S0")
    plt.show()

    plt.plot(S0, put)
    plt.xlabel("Initial asset price (S0)")
    plt.ylabel("Asian put option price")
    plt.title("Dependence of Asian Put Option on S0")
    plt.show()

  return call, put


def variation_with_K(S0, r, sigma, display=True):
  K = np.linspace(50, 150, num=250)
  call, put = [], []

  for i in K:
    call_price, put_price, _, _ = asian_option_price(S0, r, sigma, i, 500, 150, 100)
    call.append(call_price)
    put.append(put_price)
  
  if display == True:
    plt.plot(K, call)
    plt.xlabel("Strike price (K)")
    plt.ylabel("Asian call option price")
    plt.title("Dependence of Asian Call Option on K")
    plt.show()

    plt.plot(K, put)
    plt.xlabel("Strike price (K)")
    plt.ylabel("Asian put option price")
    plt.title("Dependence of Asian Put Option on K")
    plt.show()

  return call, put


def variation_with_r(S0, sigma, K, display=True):
  r = np.linspace(0, 0.5, num=120, endpoint=False)
  call, put = [], []

  for i in r:
    call_price, put_price, _, _ = asian_option_price(S0, i, sigma, K, 500, 150, 100)
    call.append(call_price)
    put.append(put_price)
  
  if display == True:
    plt.plot(r, call)
    plt.xlabel("Risk-free rate (r)")
    plt.ylabel("Asian call option price")
    plt.title("Dependence of Asian Call Option on r")
    plt.show()

    plt.plot(r, put)
    plt.xlabel("Risk-free rate (r)")
    plt.ylabel("Asian put option price")
    plt.title("Dependence of Asian Put Option on r")
    plt.show()

  return call, put


def variation_with_sigma(S0, r, K, display=True):
  sigma = np.linspace(0, 1, num=120, endpoint=False)
  call, put = [], []

  for i in sigma:
    call_price, put_price, _, _ = asian_option_price(S0, r, i, K, 500, 150, 100)
    call.append(call_price)
    put.append(put_price)
  
  if display == True:
    plt.plot(sigma, call)
    plt.xlabel("Volatility (sigma)")
    plt.ylabel("Asian call option price")
    plt.title("Dependence of Asian Call Option on sigma")
    plt.show()

    plt.plot(sigma, put)
    plt.xlabel("Volatility (sigma)")
    plt.ylabel("Asian put option price")
    plt.title("Dependence of Asian Put Option on sigma")
    plt.show()

  return call, put


def main():
  compute_path(100, 0.1, 0.2, "Asset price in real world")
  compute_path(100, 0.05, 0.2, "Asset price in risk-neutral world")

  for K in [90, 105, 110]:
    call_price, put_price, call_var, put_var = asian_option_price(100, 0.05, 0.2, K)
    print("\n\n************** For K = {} **************".format(K))
    print("Asian call option price \t\t=", call_price)
    print("Variance in Asian call option price \t=", call_var)
    print()
    print("Asian put option price \t\t\t=", put_price)
    print("Variance in Asian put option price \t=", put_var)
  
  
  # Sensitivity Analysis
  variation_with_S0(0.05, 0.2, 105)
  variation_with_K(100, 0.05, 0.2)
  variation_with_r(100, 0.2, 105)
  variation_with_sigma(100, 0.05, 105)
  

if __name__=="__main__":
  main()