"""
Kindly install these libraries before executing this code:
  1. numpy
  2. matplotlib
"""

import numpy as np
import math
import matplotlib.pyplot as plt

# if using a Jupyter notebook, kindly uncomment the following line:
# %matplotlib inline

M_list = [1, 5, 10, 20, 50, 100, 200, 400]
S = 100
K = 105
T = 5
r = 0.05
sigma = 0.3


# returns True if arbitrage opportunity exists
def arbitrage_condition(u, d, r, t):
  if d < math.exp(r*t) and math.exp(r*t) < u:
    return False
  else:
    return True


def binomial_model():
  call_option_prices = []
  put_option_prices = []

  for M in M_list:
    t = T/M
    u = math.exp(sigma*math.sqrt(t) + (r - 0.5*sigma*sigma)*t)
    d = math.exp(-sigma*math.sqrt(t) + (r - 0.5*sigma*sigma)*t)

    R = math.exp(r*t)
    p = (R - d)/(u - d);

    result = arbitrage_condition(u, d, r, t)
    if result:
      print("Arbitrage Opportunity exists for M = {}".format(M))
      call_option_prices.append(-1)
      put_option_prices.append(-1)
      continue
    else:
      print("No arbitrage exists for M = {}".format(M))

    C = [[0 for i in range(M + 1)] for j in range(M + 1)]
    P = [[0 for i in range(M + 1)] for j in range(M + 1)]

    for i in range(0, M + 1):
      C[M][i] = max(0, S*math.pow(u, M - i)*math.pow(d, i) - K)
      P[M][i] = max(0, K - S*math.pow(u, M - i)*math.pow(d, i))

    for j in range(M - 1, -1, -1):
      for i in range(0, j + 1):
        C[j][i] = (p*C[j + 1][i] + (1 - p)*C[j + 1][i + 1]) / R;
        P[j][i] = (p*P[j + 1][i] + (1 - p)*P[j + 1][i + 1]) / R;

    call_option_prices.append(C[0][0])
    put_option_prices.append(P[0][0])
  
  return call_option_prices, put_option_prices


def main():
  call_option_prices, put_option_prices = binomial_model()
  print()
  for idx in range(0, len(M_list)):
    print("M = {}\t\tCall Option = {:.5f}\t\tPut Option = {:.5f}".format(M_list[idx], call_option_prices[idx], put_option_prices[idx]))
  

if __name__=="__main__":
  main()