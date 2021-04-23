"""
Kindly install these libraries before executing this code:
  1. numpy
  2. matplotlib
  3. scipy
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D 

# if using a Jupyter notebook, kindly uncomment the following line:
# %matplotlib inline


def BSM_model(x, t, T, K, r, sigma):
  if t == T:
    return max(0, x - K), max(0, K - x)

  d1 = ( math.log(x/K) + (r + 0.5 * sigma * sigma) * (T - t) ) / ( sigma * math.sqrt(T - t) )
  d2 = ( math.log(x/K) + (r - 0.5 * sigma * sigma) * (T - t) ) / ( sigma * math.sqrt(T - t) )

  call_price = x * norm.cdf(d1) - K * math.exp( -r * (T - t) ) * norm.cdf(d2)
  put_price = K * math.exp( -r * (T - t) ) * norm.cdf(-d2) - x * norm.cdf(-d1)

  return call_price, put_price


def plot_surface(T, K, r, sigma):
  call_prices_list, put_prices_list = [], []
  x_list = np.linspace(0.0001, 2, num = 100)
  t_list = np.linspace(0, 1, num = 100)

  x_list, t_list = np.meshgrid(x_list, t_list)
  row, col = len(x_list), len(x_list[0])

  for i in range(row):
    call_prices_list.append([])
    put_prices_list.append([])

    for j in range(col):
      C, P = BSM_model(x_list[i][j], t_list[i][j], T, K, r, sigma)
      call_prices_list[i].append(C)
      put_prices_list[i].append(P)

  call_prices_list = np.array(call_prices_list)
  put_prices_list = np.array(put_prices_list)  
  
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(x_list, t_list, call_prices_list, cmap=cm.coolwarm)
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.title('C(t, x) vs x and t')
  ax.set_xlabel("x") 
  ax.set_ylabel("t") 
  ax.set_zlabel("C(t, x")
  plt.show()

  fig = plt.figure()
  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(x_list, t_list, put_prices_list, cmap=cm.coolwarm)
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.title('P(t, x) vs x and t')
  ax.set_xlabel("x") 
  ax.set_ylabel("t") 
  ax.set_zlabel("P(t, x")
  plt.show()
  

def main():
  plot_surface(1, 1, 0.05, 0.6)
      

if __name__=="__main__":
  main()