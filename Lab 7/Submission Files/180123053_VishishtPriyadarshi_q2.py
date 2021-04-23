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
from mpl_toolkits import mplot3d

# if using a Jupyter notebook, kindly uncomment the following line:
# %matplotlib inline


def plot_3D(x_list, t_list, call_prices, z_label, plt_title):
  x, y, z = [], [], []

  for idx1 in range(len(t_list)):
    for idx2 in range(len(x_list)):
      x.append(x_list[idx2])
      y.append(t_list[idx1])
      z.append(call_prices[idx1][idx2])
    
  ax = plt.axes(projection='3d')
  ax.scatter3D(x, y, z, cmap='Greens')
  plt.title(plt_title)
  ax.set_xlabel("x") 
  ax.set_ylabel("t") 
  ax.set_zlabel(z_label)
  plt.show()


def BSM_model(x, t, T, K, r, sigma):
  if t == T:
    return max(0, x - K), max(0, K - x)

  d1 = ( math.log(x/K) + (r + 0.5 * sigma * sigma) * (T - t) ) / ( sigma * math.sqrt(T - t) )
  d2 = ( math.log(x/K) + (r - 0.5 * sigma * sigma) * (T - t) ) / ( sigma * math.sqrt(T - t) )

  call_price = x * norm.cdf(d1) - K * math.exp( -r * (T - t) ) * norm.cdf(d2)
  put_price = K * math.exp( -r * (T - t) ) * norm.cdf(-d2) - x * norm.cdf(-d1)

  return call_price, put_price


def plot_graphs(t_list, T, K, r, sigma):
  call_prices_list, put_prices_list = [], []
  x_list = np.linspace(0.1, 2, num = 1000)

  for t in t_list:
    call_prices, put_prices = [], []  

    for x in x_list:
      C, P = BSM_model(x, t, T, K, r, sigma)
      call_prices.append(C)
      put_prices.append(P)
    
    call_prices_list.append(call_prices)
    put_prices_list.append(put_prices)
  
  for idx in range(len(t_list)):
    plt.plot(x_list, call_prices_list[idx], label = 't = {}'.format(t_list[idx]))
  plt.xlabel('x')
  plt.ylabel('C(t, x)')
  plt.title('Plot for C(t,x) vs x')
  plt.legend()
  plt.grid()
  plt.show()

  plot_3D(x_list, t_list, call_prices_list, "C(t, x)", "Dependence of C(t, x) on t and x")
  

  for idx in range(len(t_list)):
    plt.plot(x_list, put_prices_list[idx], label = 't = {}'.format(t_list[idx]))
  plt.xlabel('x')
  plt.ylabel('P(t, x)')
  plt.title('Plot for P(t,x) vs x')
  plt.legend()
  plt.grid()
  plt.show()

  plot_3D(x_list, t_list, put_prices_list, "P(t, x)", "Dependence of P(t, x) on t and x")


def main():
  t = [0, 0.2, 0.4, 0.6, 0.8, 1]
  plot_graphs(t, 1, 1, 0.05, 0.6)
      

if __name__=="__main__":
  main()