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


def vasicek_model(beta, mu, sigma, r, t, T_list):
  Yield = []

  for T in T_list:  
    B = (1 - math.exp(-beta *  (T - t))) / beta
    A = math.exp((B - T + t)*(beta*beta*mu - sigma*sigma*0.5)/(beta*beta) - math.pow(sigma * B, 2)/(4*beta))
    P = A * math.exp(-B * r)
    y = -math.log(P) / (T - t)
    Yield.append(y)
  
  return Yield


def main():
  values = [[5.9, 0.2, 0.3, 0.1], [3.9, 0.1, 0.3, 0.2], [0.1, 0.4, 0.11, 0.1]]
  for idx in range(len(values)):
    beta, mu, sigma, r = values[idx]
    T = np.linspace(0.01, 10, num=10, endpoint=False)
    Yield = vasicek_model(beta, mu, sigma, r, 0, T)

    plt.plot(T, Yield, marker='o')
    plt.xlabel('Maturity (T)')
    plt.ylabel('Yield')
    plt.title('Term structure for parameter set - {}'.format(idx + 1))
    plt.show()
  
  T = np.linspace(0.01, 10, num=500, endpoint=False)
  r_list = [0.1 * i for i in range(1, 11)]
  for idx in range(len(values)):
    beta, mu, sigma, r = values[idx]
    for r in r_list:
      Yield = vasicek_model(beta, mu, sigma, r, 0, T)
      plt.plot(T, Yield)

    plt.xlabel('Maturity (T)')
    plt.ylabel('Yield')
    print("************* Parameter set - {} *************".format(idx + 1))
    plt.title('Term structure for 10 different values of r(0) & 500 time units'.format(idx + 1))
    plt.show()


if __name__=="__main__":
  main()