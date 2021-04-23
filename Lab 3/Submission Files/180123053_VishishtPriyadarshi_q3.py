"""
Kindly install these libraries before executing this code:
  1. numpy
  2. matplotlib
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from functools import reduce
import operator as op
import time


# if using a Jupyter notebook, kindly uncomment the following line:
# %matplotlib inline


def nCr(n, r):
  r = min(r, n-r)
  numer = reduce(op.mul, range(n, n-r, -1), 1)
  denom = reduce(op.mul, range(1, r+1), 1)
  return numer // denom  
                           

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
  
  return S0


def binomial_method_efficient(S0, K, T, M, r, sigma, display):
  curr_time = time.time()
  t = T/M
  u = math.exp(sigma*math.sqrt(t) + (r - 0.5*sigma*sigma)*t)
  d = math.exp(-sigma*math.sqrt(t) + (r - 0.5*sigma*sigma)*t)

  R = math.exp(r*t)
  p = (R - d)/(u - d);

  result = arbitrage_condition(u, d, r, t)
  if result:
    if display == 1:
      print("Arbitrage Opportunity exists for M = {}".format(M))
    return
  else:
    if display == 1:
      print("No arbitrage exists for M = {}".format(M))

  C = [[0 for i in range(M + 1)] for j in range(M + 1)]

  for i in range(0, M + 1):
    C[M][i] = max(0, S0*math.pow(u, M - i)*math.pow(d, i) - K)

  for j in range(M - 1, -1, -1):
    for i in range(0, j + 1):
      C[j][i] = (p*C[j + 1][i] + (1 - p)*C[j + 1][i + 1]) / R;
    
  if display == 1: 
    print("European Call Option \t\t= {}".format(C[0][0]))
    print("Execution Time \t\t\t= {} sec\n".format(time.time() - curr_time))

  if display == 2:
    for i in range(M + 1):
      print("At t = {}".format(i))
      for j in range(i + 1):
        print("Index no = {}\tPrice = {}".format(j, C[i][j]))
      print()
      
  return C[0][0]


def binomial_method_most_efficient(S0, K, T, M, r, sigma, display):
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

  option_price = 0
  for j in range(0, M + 1):
    option_price += nCr(M, j) * math.pow(p, j) * math.pow(1 - p, M - j) * max(S0 * math.pow(u, j) * math.pow(d, M - j) - K, 0)
  
  option_price /= math.pow(R, M)


  if display == 1: 
    print("European Call Option \t\t= {}".format(option_price))
    print("Execution Time \t\t\t= {} sec\n".format(time.time() - curr_time))


  return option_price


def binomial_method_unoptimised(S0, K, T, M, r, sigma, display):
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
    option_price[M][i] = max(req_price - K, 0)
  
  for j in range(M - 1, -1, -1):
    for i in range(0, int(pow(2, j))):
      option_price[j][i] = (p*option_price[j + 1][2*i] + (1 - p)*option_price[j + 1][2*i + 1]) / R;

  if display == 1: 
    print("European Call Option \t\t= {}".format(option_price[0][0]))
    print("Execution Time \t\t\t= {} sec\n".format(time.time() - curr_time))
    
  return option_price[0][0]


def main():
  # sub-part (a)
  print("-----------------------  sub-part(a)  -----------------------\n")
  M1 = [5, 10, 25]
  M2 = [5, 10, 25, 50]
  prices1, prices2, prices3 = [], [], []
 
  print('######################  Unoptimised Binomial Algorithm executing  ######################')
  for m in M1:
    prices1.append(binomial_method_unoptimised(S0 = 100, T = 1, K = 100, M = m, r = 0.08, sigma = 0.20, display = 1))


  print('\n\n######################  Efficient Binomial Algorithm executing (Markov Based)  ######################')
  for m in M2:
    prices2.append(binomial_method_efficient(S0 = 100, T = 1, K = 100, M = m, r = 0.08, sigma = 0.20, display = 1))


  print('\n\n######################  Most Efficient Binomial Algorithm executing (Markov Based) ######################')
  for m in M2:
    prices3.append(binomial_method_most_efficient(S0 = 100, T = 1, K = 100, M = m, r = 0.08, sigma = 0.20, display = 1))


  # sub-part (b)
  print("\n\n-----------------------  sub-part(b)  -----------------------")
  plot_fixed(M2, prices2, "M", "Initial Option Prices", "Initial Option Prices vs M")
  M = [i for i in range(1, 21)]
  prices1.clear()
  for m in M:
    prices1.append(binomial_method_most_efficient(S0 = 100, T = 1, K = 100, M = m, r = 0.08, sigma = 0.20, display = 0))

  plot_fixed(M, prices1, "M", "Initial Option Prices", "Initial Option Prices vs M (Variation with more data-points for M)")


  # sub-part (c)
  print("\n\n-----------------------  sub-part(c)  -----------------------")
  binomial_method_efficient(S0 = 100, T = 1, K = 100, M = 5, r = 0.08, sigma = 0.20, display = 2)


if __name__=="__main__":
  main()