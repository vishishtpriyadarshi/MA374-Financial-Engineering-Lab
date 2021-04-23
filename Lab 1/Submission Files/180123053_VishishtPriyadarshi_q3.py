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

time_points = [0, 0.50, 1, 1.50, 3, 4.5]
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


def check_time_stamp(t):
  for i in time_points:
    if t * 0.25 == i:
      return True

  return False


def binomial_model():
  call_option_prices = []
  put_option_prices = []
  M = 20
  t = T/M
  u = math.exp(sigma*math.sqrt(t) + (r - 0.5*sigma*sigma)*t)
  d = math.exp(-sigma*math.sqrt(t) + (r - 0.5*sigma*sigma)*t)

  R = math.exp(r*t)
  p = (R - d)/(u - d);

  result = arbitrage_condition(u, d, r, t)
  if result:
    # print("Arbitrage Opportunity exists for M = {}".format(M))
    call_option_prices.append(-1)
    put_option_prices.append(-1)
    return
  # else:
  #   print("No arbitrage exists for M = {}".format(M))

  C = [[0 for i in range(M + 1)] for j in range(M + 1)]
  P = [[0 for i in range(M + 1)] for j in range(M + 1)]

  for i in range(0, M + 1):
    C[M][i] = max(0, S*math.pow(u, M - i)*math.pow(d, i) - K)
    P[M][i] = max(0, K - S*math.pow(u, M - i)*math.pow(d, i))

  for j in range(M - 1, -1, -1):
    for i in range(0, j + 1):
      C[j][i] = (p*C[j + 1][i] + (1 - p)*C[j + 1][i + 1]) / R;
      P[j][i] = (p*P[j + 1][i] + (1 - p)*P[j + 1][i + 1]) / R;

  for i in range(0, M + 1, 2):
    if check_time_stamp(i):
      intermediate_call = []
      intermediate_put = []

      for j in range(0, i + 1):
        intermediate_call.append(C[i][j])
        intermediate_put.append(P[i][j])
      
      call_option_prices.append(intermediate_call)
      put_option_prices.append(intermediate_put)

  return call_option_prices, put_option_prices


def main():
  call_option_prices, put_option_prices = binomial_model()
  for idx in range(0, len(time_points)):
    print("t = {}".format(time_points[idx]))
    print("Call Option\tPut Option")
    for j in range(len(call_option_prices[idx])):
      print("{:.2f}\t\t{:.2f}".format(call_option_prices[idx][j], put_option_prices[idx][j]))
    
    print()


if __name__=="__main__":
  main()