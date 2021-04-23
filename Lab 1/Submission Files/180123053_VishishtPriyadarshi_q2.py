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

step_size = [1, 5]
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


def plot(call, put, step):
  x = np.arange(1, 401, step)
  plt.plot(x, call)
  plt.xlabel("Number of subintervals (M)")
  plt.ylabel("Prices of Call option at t = 0") 
  plt.title("Initial Call Option Price vs M    (for the step value = {})".format(step))
  plt.show()

  plt.plot(x, put)
  plt.xlabel("Number of subintervals (M)")
  plt.ylabel("Prices of Put option at t = 0") 
  plt.title("Initial Put Option Price vs M    (for the step value = {})".format(step))
  plt.show()
  

def binomial_model():
  for step in step_size:
    call_option_prices = []
    put_option_prices = []
    for M in range(1, 401, step):
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
        continue
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

      call_option_prices.append(C[0][0])
      put_option_prices.append(P[0][0])

    plot(call_option_prices, put_option_prices, step)


def main():
  binomial_model()


if __name__=="__main__":
  main()