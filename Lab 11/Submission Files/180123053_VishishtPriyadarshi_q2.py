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


def cir_model(beta, mu, sigma, r, t, T_list):
  Yield = []

  for T in T_list:  
    gamma = math.sqrt(beta*beta + 2*sigma*sigma)
    B = 2 *(math.exp(gamma * (T - t)) - 1)  / ( 2*gamma + (gamma + beta) * (math.exp(gamma * (T - t)) - 1))
    A = math.pow( ( 2*gamma*math.exp(0.5*(beta + gamma)*(T - t)) ) / (2*gamma + (gamma + beta)*(math.exp(gamma*(T - t)) - 1)), 2*beta*mu / (sigma*sigma) ) 
    P = A * math.exp(-B * r)
    y = -math.log(P) / (T - t)
    Yield.append(y)
  
  return Yield


def main():
  values = [[0.02, 0.7, 0.02, 0.1], [0.7, 0.1, 0.3, 0.2], [0.06, 0.09, 0.5, 0.02]]
  for idx in range(len(values)):
    beta, mu, sigma, r = values[idx]
    T = np.linspace(0.1, 10, num=10, endpoint=False)
    Yield = cir_model(beta, mu, sigma, r, 0, T)

    plt.plot(T, Yield, marker='o')
    plt.xlabel('Maturity (T)')
    plt.ylabel('Yield')
    plt.title('Term structure for parameter set - {}'.format(idx + 1))
    plt.show()
  
  T = np.linspace(0.1, 600, num=600, endpoint=False)
  r_list = [0.1 * i for i in range(1, 11)]
  values = [[0.02, 0.7, 0.02]]
  for idx in range(len(values)):
    beta, mu, sigma = values[idx]
    for r in r_list:
      Yield = cir_model(beta, mu, sigma, r, 0, T)
      plt.plot(T, Yield)
    plt.xlabel('Maturity (T)')
    plt.ylabel('Yield')
    plt.title('Term structure for 10 different values of r(0) & 600 time units'.format(idx + 1))
    plt.show()


if __name__=="__main__":
  main()