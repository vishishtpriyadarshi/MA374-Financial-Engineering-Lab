"""
Kindly install these libraries before executing this code:
  1. numpy
  2. matplotlib
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import time

# if using a Jupyter notebook, kindly uncomment the following line:
# %matplotlib inline


def plot_fixed(x, y, x_axis, y_axis, title):
  plt.plot(x, y)
  plt.xlabel(x_axis)
  plt.ylabel(y_axis) 
  plt.title(title)
  plt.show()


# returns True if arbitrage opportunity exists
def arbitrage_condition(u, d, r, t):
  if d < math.exp(r*t) and math.exp(r*t) < u:
    return False
  else:
    return True


def cache(idx, u, d, p, R, M, stock_price, running_max, option_prices):
  if idx == M + 1 or (stock_price, running_max) in option_prices[idx]:
    return

  cache(idx + 1, u, d, p, R, M, stock_price*u, max(stock_price*u, running_max), option_prices)
  cache(idx + 1, u, d, p, R, M, stock_price*d, max(stock_price*d, running_max), option_prices)

  if idx == M:
    option_prices[M][(stock_price, running_max)] = max(running_max - stock_price, 0)
  else:
    option_prices[idx][(stock_price, running_max)] = (p*option_prices[idx + 1][ (u * stock_price, max(u * stock_price, running_max)) ] + (1 - p)*option_prices[idx + 1][ (d * stock_price, running_max) ]) / R
  

def loopback_option_efficient(S0, T, M, r, sigma, display):
  if display == 1: 
    print("\n\n*********  Executing for M = {}  *********\n".format(M))
  curr_time_1 = time.time()

  u, d = 0, 0
  t = T/M

  u = math.exp(sigma*math.sqrt(t) + (r - 0.5*sigma*sigma)*t)
  d = math.exp(-sigma*math.sqrt(t) + (r - 0.5*sigma*sigma)*t)  

  R = math.exp(r*t)
  p = (R - d)/(u - d);
  result = arbitrage_condition(u, d, r, t)

  option_prices = []
  for i in range(0, M + 1):
    option_prices.append(dict())

  cache(0, u, d, p, R, M, S0, S0, option_prices)
  
  if result:
    if display == 1:
      print("Arbitrage Opportunity exists for M = {}".format(M))
    return 0, 0
  else:
    if display == 1:
      print("No arbitrage exists for M = {}".format(M))

  if display == 1: 
    print("Initial Price of Loopback Option \t= {}".format(option_prices[0][ (S0, S0) ]))
    print("Execution Time \t\t\t\t= {} sec\n".format(time.time() - curr_time_1))
  
  if display == 2:
    for i in range(len(option_prices)):
      print("At t = {}".format(i))
      for key, value in option_prices[i].items():
        print("Intermediate state = {}\t\tPrice = {}".format(key, value))
      print()

  return option_prices[0][ (S0, S0) ]


def main():
  # sub-part (a)
  print("-----------------------  sub-part(a)  -----------------------")
  M = [5, 10, 25, 50]
  prices = []

  for m in M:
    prices.append(loopback_option_efficient(S0 = 100, T = 1, M = m, r = 0.08, sigma = 0.20, display = 1))
  

  # sub-part (b)
  print("\n\n-----------------------  sub-part(b)  -----------------------")
  plot_fixed(M, prices, "M", "Initial Option Prices", "Initial Option Prices vs M")
  M = [i for i in range(1, 21)]
  prices.clear()
  for m in M:
    prices.append(loopback_option_efficient(S0 = 100, T = 1, M = m, r = 0.08, sigma = 0.20, display = 0))

  plot_fixed(M, prices, "M", "Initial Option Prices", "Initial Option Prices vs M (Variation with more data-points for M)")


  # sub-part (c)
  print("\n\n-----------------------  sub-part(c)  -----------------------")
  loopback_option_efficient(S0 = 100, T = 1, M = 5, r = 0.08, sigma = 0.20, display = 2)


if __name__=="__main__":
  main()