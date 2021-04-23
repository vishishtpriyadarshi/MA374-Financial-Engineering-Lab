"""
Kindly install these libraries before executing this code:
  1. numpy
  2. matplotlib
  3. scipy
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm


# if using a Jupyter notebook, kindly uncomment the following line:
# %matplotlib inline


def BSM_model(x, t, T, K, r, sigma):
  d1 = ( math.log(x/K) + (r + 0.5 * sigma * sigma) * (T - t) ) / ( sigma * math.sqrt(T - t) )
  d2 = ( math.log(x/K) + (r - 0.5 * sigma * sigma) * (T - t) ) / ( sigma * math.sqrt(T - t) )
  

  call_price = x * norm.cdf(d1) - K * math.exp( -r * (T - t) ) * norm.cdf(d2)
  put_price = K * math.exp( -r * (T - t) ) * norm.cdf(-d2) - x * norm.cdf(-d1)

  return call_price, put_price


def main():
  C, P = BSM_model(1.5, 0, 1, 1, 0.05, 0.6)
  print("Using Model paramaters -\nx = {}\nt = {}\nT = {}\nK = {}\nr = {}\nsigma = {}\n".format(1.5, 0, 1, 1, 0.05, 0.6))
  print("European Call Price =", C)
  print("European Put Price =", P)
      

if __name__=="__main__":
  main()