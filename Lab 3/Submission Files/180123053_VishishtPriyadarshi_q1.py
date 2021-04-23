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


def compute_option_price(i, S0, u, d, M):
  path = format(i, 'b').zfill(M)
  curr_max = S0

  for idx in path:
    if idx == '1':
      S0 *= d
    else:
      S0 *= u

    curr_max = max(curr_max, S0)
  
  return curr_max - S0


def loopback_option_basic(S0, T, M, r, sigma, display):
  if display == 1: 
    print("\n\n*********  Executing for M = {}  *********\n".format(M))
  curr_time = time.time()
  
  u, d = 0, 0
  t = T/M
  u = math.exp(sigma*math.sqrt(t) + (r - 0.5*sigma*sigma)*t)
  d = math.exp(-sigma*math.sqrt(t) + (r - 0.5*sigma*sigma)*t)  

  R = math.exp(r*t)
  p = (R - d)/(u - d);
  result = arbitrage_condition(u, d, r, t)

  if result:
    if display == 1:
      print("Arbitrage Opportunity exists for M = {}".format(M))
    return 0, 0
  else:
    if display == 1:
      print("No arbitrage exists for M = {}".format(M))

  option_price = []
  for i in range(0, M + 1):
    D = []
    for j in range(int(pow(2, i))):
      D.append(0)
    option_price.append(D)
    
  for i in range(int(pow(2, M))):
    req_price = compute_option_price(i, S0, u, d, M)
    option_price[M][i] = max(req_price, 0)
  
  for j in range(M - 1, -1, -1):
    for i in range(0, int(pow(2, j))):
      option_price[j][i] = (p*option_price[j + 1][2*i] + (1 - p)*option_price[j + 1][2*i + 1]) / R;

  if display == 1: 
    print("Initial Price of Loopback Option \t= {}".format(option_price[0][0]))
    print("Execution Time \t\t\t\t= {} sec\n".format(time.time() - curr_time))

  if display == 2:
    for i in range(len(option_price)):
      print("At t = {}".format(i))
      for j in range(len(option_price[i])):
        print("Index no = {}\tPrice = {}".format(j, option_price[i][j]))
      print()
    
  return option_price[0][0]


def main():
  # sub-part (a)
  print("-----------------------  sub-part(a)  -----------------------")
  M = [5, 10, 25]
  prices = []
  for m in M:
    prices.append(loopback_option_basic(S0 = 100, T = 1, M = m, r = 0.08, sigma = 0.20, display = 1))


  # sub-part (b)
  print("\n\n-----------------------  sub-part(b)  -----------------------")
  plot_fixed(M, prices, "M", "Initial Option Prices", "Initial Option Prices vs M")
  M = [i for i in range(1, 21)]
  prices.clear()
  for m in M:
    prices.append(loopback_option_basic(S0 = 100, T = 1, M = m, r = 0.08, sigma = 0.20, display = 0))

  plot_fixed(M, prices, "M", "Initial Option Prices", "Initial Option Prices vs M (Variation with more data-points for M)")


  # sub-part (c)
  print("\n\n-----------------------  sub-part(c)  -----------------------")
  loopback_option_basic(S0 = 100, T = 1, M = 5, r = 0.08, sigma = 0.20, display = 2)


if __name__=="__main__":
  main()