"""
Kindly install these libraries before executing this code:
  1. numpy
  2. matplotlib
  3. scipy
  4. tabulate
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D 
from tabulate import tabulate

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


def variation_with_T(T_sample, K, r, sigma):
  t = 0
  x_list = [0.4, 0.6, 0.8, 1.0, 1.2]

  T_list = np.linspace(0.1, 5, num = 500)
  call_prices_list, put_prices_list = [], []

  counter = 0
  data = []

  for x in x_list:
    call, put = [], []
    for T in T_list:
      C, P = BSM_model(x, t, T, K, r, sigma)
      call.append(C)
      put.append(P)
  
      if x == 0.8:
          if counter % 50 == 0:
            data.append([1 + int(counter/50), T, C, P])
          counter += 1

    call_prices_list.append(call)
    put_prices_list.append(put)
  
  print("*********************** Variation of C(t, x) and P(t, x) with T ***********************\n")
  heading = ['SI No', 'T', 'C(t, x)', 'P(t, x)']
  print(tabulate(data, headers = heading))
  
  for idx in range(len(x_list)):
    plt.plot(T_list, call_prices_list[idx], label = 'x = {}'.format(x_list[idx]))

  plt.xlabel('T')
  plt.ylabel('C(t, x)')
  plt.title('Plot for C(t,x) vs T')
  plt.legend()
  plt.grid()
  plt.show()

  for idx in range(len(x_list)):
    plt.plot(T_list, put_prices_list[idx], label = 'x = {}'.format(x_list[idx]))

  plt.xlabel('T')
  plt.ylabel('P(t, x)')
  plt.title('Plot for P(t,x) vs T')
  plt.legend()
  plt.grid()
  plt.show()


def variation_with_K(T, K_sample, r, sigma):
  t = 0
  x_list = [0.4, 0.6, 0.8, 1.0, 1.2]

  K_list = np.linspace(0.1, 2, num = 500)
  call_prices_list, put_prices_list = [], []

  counter = 0
  data = []

  for x in x_list:
    call, put = [], []
    for K in K_list:
      C, P = BSM_model(x, t, T, K, r, sigma)
      call.append(C)
      put.append(P)

      if x == 0.8:
        if counter % 50 == 0:
          data.append([1 + int(counter/50), K, C, P])
        counter += 1

    call_prices_list.append(call)
    put_prices_list.append(put)
  
  print("*********************** Variation of C(t, x) and P(t, x) with K ***********************\n")
  heading = ['SI No', 'K', 'C(t, x)', 'P(t, x)']
  print(tabulate(data, headers = heading))
  
  
  for idx in range(len(x_list)):
    plt.plot(K_list, call_prices_list[idx], label = 'x = {}'.format(x_list[idx]))

  plt.xlabel('K')
  plt.ylabel('C(t, x)')
  plt.title('Plot for C(t,x) vs K')
  plt.legend()
  plt.grid()
  plt.show()

  for idx in range(len(x_list)):
    plt.plot(K_list, put_prices_list[idx], label = 'x = {}'.format(x_list[idx]))

  plt.xlabel('K')
  plt.ylabel('P(t, x)')
  plt.title('Plot for P(t,x) vs K')
  plt.legend()
  plt.grid()
  plt.show()


def variation_with_r(T, K, r_sample, sigma):
  t = 0
  x_list = [0.4, 0.6, 0.8, 1.0, 1.2]

  r_list = np.linspace(0, 1, num = 500, endpoint = False)
  call_prices_list, put_prices_list = [], []

  counter = 0
  data = []

  for x in x_list:
    call, put = [], []
    for r in r_list:
      C, P = BSM_model(x, t, T, K, r, sigma)
      call.append(C)
      put.append(P)
    
      if x == 0.8:
          if counter % 50 == 0:
            data.append([1 + int(counter/50), r, C, P])
          counter += 1

    call_prices_list.append(call)
    put_prices_list.append(put)
  
  print("*********************** Variation of C(t, x) and P(t, x) with r ***********************\n")
  heading = ['SI No', 'r', 'C(t, x)', 'P(t, x)']
  print(tabulate(data, headers = heading))
  
  for idx in range(len(x_list)):
    plt.plot(r_list, call_prices_list[idx], label = 'x = {}'.format(x_list[idx]))

  plt.xlabel('r')
  plt.ylabel('C(t, x)')
  plt.title('Plot for C(t,x) vs r')
  plt.legend()
  plt.grid()
  plt.show()

  for idx in range(len(x_list)):
    plt.plot(r_list, put_prices_list[idx], label = 'x = {}'.format(x_list[idx]))

  plt.xlabel('r')
  plt.ylabel('P(t, x)')
  plt.title('Plot for P(t,x) vs r')
  plt.legend()
  plt.grid()
  plt.show()


def variation_with_sigma(T, K, r, sigma_sample):
  t = 0
  x_list = [0.4, 0.6, 0.8, 1.0, 1.2]

  sigma_list = np.linspace(0.001, 1, num = 500, endpoint = False)
  call_prices_list, put_prices_list = [], []

  counter = 0
  data = []

  for x in x_list:
    call, put = [], []
    for sigma in sigma_list:
      C, P = BSM_model(x, t, T, K, r, sigma)
      call.append(C)
      put.append(P)
    
      if x == 0.8:
          if counter % 50 == 0:
            data.append([1 + int(counter/50), sigma, C, P])
          counter += 1

    call_prices_list.append(call)
    put_prices_list.append(put)
  
  print("*********************** Variation of C(t, x) and P(t, x) with sigma ***********************\n")
  heading = ['SI No', 'sigma', 'C(t, x)', 'P(t, x)']
  print(tabulate(data, headers = heading))
  
  for idx in range(len(x_list)):
    plt.plot(sigma_list, call_prices_list[idx], label = 'x = {}'.format(x_list[idx]))

  plt.xlabel('sigma')
  plt.ylabel('C(t, x)')
  plt.title('Plot for C(t,x) vs sigma')
  plt.legend()
  plt.grid()
  plt.show()

  for idx in range(len(x_list)):
    plt.plot(sigma_list, put_prices_list[idx], label = 'x = {}'.format(x_list[idx]))

  plt.xlabel('sigma')
  plt.ylabel('P(t, x)')
  plt.title('Plot for P(t,x) vs sigma')
  plt.legend()
  plt.grid()
  plt.show()


def variation_with_K_and_r(x, t, T, sigma):
  print("*********************** Variation of C(t, x) and P(t, x) with K and r ***********************\n")
  call_prices_list, put_prices_list = [], []
  K_list = np.linspace(0.01, 2, num = 100)
  r_list = np.linspace(0, 1, num = 100, endpoint = False)

  K_list, r_list = np.meshgrid(K_list, r_list)
  row, col = len(K_list), len(K_list[0])

  for i in range(row):
    call_prices_list.append([])
    put_prices_list.append([])

    for j in range(col):
      C, P = BSM_model(x, t, T, K_list[i][j], r_list[i][j], sigma)
      call_prices_list[i].append(C)
      put_prices_list[i].append(P)

  call_prices_list = np.array(call_prices_list)
  put_prices_list = np.array(put_prices_list)  
  
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(K_list, r_list, call_prices_list, cmap=cm.coolwarm)
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.title('C(t, x) vs K and r')
  ax.set_xlabel("K") 
  ax.set_ylabel("r") 
  ax.set_zlabel("C(t, x")
  plt.show()

  fig = plt.figure()
  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(K_list, r_list, put_prices_list, cmap=cm.coolwarm)
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.title('P(t, x) vs K and r')
  ax.set_xlabel("K") 
  ax.set_ylabel("r") 
  ax.set_zlabel("P(t, x")
  plt.show()


def variation_with_K_and_sigma(x, t, T, r):
  print("*********************** Variation of C(t, x) and P(t, x) with K and sigma ***********************\n")
  call_prices_list, put_prices_list = [], []
  K_list = np.linspace(0.01, 2, num = 100)
  sigma_list = np.linspace(0.01, 1, num = 100, endpoint = False)

  K_list, sigma_list = np.meshgrid(K_list, sigma_list)
  row, col = len(K_list), len(K_list[0])

  for i in range(row):
    call_prices_list.append([])
    put_prices_list.append([])

    for j in range(col):
      C, P = BSM_model(x, t, T, K_list[i][j], r, sigma_list[i][j])
      call_prices_list[i].append(C)
      put_prices_list[i].append(P)

  call_prices_list = np.array(call_prices_list)
  put_prices_list = np.array(put_prices_list)  
  
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(K_list, sigma_list, call_prices_list, cmap=cm.coolwarm)
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.title('C(t, x) vs K and sigma')
  ax.set_xlabel("K") 
  ax.set_ylabel("sigma") 
  ax.set_zlabel("C(t, x")
  plt.show()

  fig = plt.figure()
  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(K_list, sigma_list, put_prices_list, cmap=cm.coolwarm)
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.title('P(t, x) vs K and sigma')
  ax.set_xlabel("K") 
  ax.set_ylabel("sigma") 
  ax.set_zlabel("P(t, x")
  plt.show()


def variation_with_r_and_sigma(x, t, T, K):
  print("*********************** Variation of C(t, x) and P(t, x) with r and sigma ***********************\n")
  call_prices_list, put_prices_list = [], []
  sigma_list = np.linspace(0.01, 1, num = 100, endpoint = False)
  r_list = np.linspace(0.001, 1, num = 100, endpoint = False)

  sigma_list, r_list = np.meshgrid(sigma_list, r_list)
  row, col = len(sigma_list), len(sigma_list[0])

  for i in range(row):
    call_prices_list.append([])
    put_prices_list.append([])

    for j in range(col):
      C, P = BSM_model(x, t, T, K, r_list[i][j], sigma_list[i][j])
      call_prices_list[i].append(C)
      put_prices_list[i].append(P)

  call_prices_list = np.array(call_prices_list)
  put_prices_list = np.array(put_prices_list)  
  
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(sigma_list, r_list, call_prices_list, cmap=cm.coolwarm)
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.title('C(t, x) vs sigma and r')
  ax.set_xlabel("sigma") 
  ax.set_ylabel("r") 
  ax.set_zlabel("C(t, x")
  plt.show()

  fig = plt.figure()
  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(sigma_list, r_list, put_prices_list, cmap=cm.coolwarm)
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.title('P(t, x) vs sigma and r')
  ax.set_xlabel("sigma") 
  ax.set_ylabel("r") 
  ax.set_zlabel("P(t, x")
  plt.show()


def variation_with_T_and_K(x, t, r, sigma):
  print("*********************** Variation of C(t, x) and P(t, x) with T and K ***********************\n")
  call_prices_list, put_prices_list = [], []
  K_list = np.linspace(0.01, 2, num = 100)
  T_list = np.linspace(0.1, 5, num = 100)

  K_list, T_list = np.meshgrid(K_list, T_list)
  row, col = len(K_list), len(K_list[0])

  for i in range(row):
    call_prices_list.append([])
    put_prices_list.append([])

    for j in range(col):
      C, P = BSM_model(x, t, T_list[i][j], K_list[i][j], r, sigma)
      call_prices_list[i].append(C)
      put_prices_list[i].append(P)

  call_prices_list = np.array(call_prices_list)
  put_prices_list = np.array(put_prices_list)  
  
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(K_list, T_list, call_prices_list, cmap=cm.coolwarm)
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.title('C(t, x) vs K and T')
  ax.set_xlabel("K") 
  ax.set_ylabel("T") 
  ax.set_zlabel("C(t, x")
  plt.show()

  fig = plt.figure()
  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(K_list, T_list, put_prices_list, cmap=cm.coolwarm)
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.title('P(t, x) vs K and T')
  ax.set_xlabel("K") 
  ax.set_ylabel("T") 
  ax.set_zlabel("P(t, x")
  plt.show()


def variation_with_T_and_r(x, t, K, sigma):
  print("*********************** Variation of C(t, x) and P(t, x) with T and r ***********************\n")
  call_prices_list, put_prices_list = [], []
  r_list = np.linspace(0.01, 1, num = 100)
  T_list = np.linspace(0.1, 5, num = 100)

  r_list, T_list = np.meshgrid(r_list, T_list)
  row, col = len(r_list), len(r_list[0])

  for i in range(row):
    call_prices_list.append([])
    put_prices_list.append([])

    for j in range(col):
      C, P = BSM_model(x, t, T_list[i][j], K, r_list[i][j], sigma)
      call_prices_list[i].append(C)
      put_prices_list[i].append(P)

  call_prices_list = np.array(call_prices_list)
  put_prices_list = np.array(put_prices_list)  
  
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(T_list, r_list, call_prices_list, cmap=cm.coolwarm)
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.title('C(t, x) vs T and r')
  ax.set_xlabel("T") 
  ax.set_ylabel("r") 
  ax.set_zlabel("C(t, x")
  plt.show()

  fig = plt.figure()
  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(T_list, r_list, put_prices_list, cmap=cm.coolwarm)
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.title('P(t, x) vs T and r')
  ax.set_xlabel("T") 
  ax.set_ylabel("r") 
  ax.set_zlabel("P(t, x")
  plt.show()


def variation_with_T_and_sigma(x, t, K, r):
  print("*********************** Variation of C(t, x) and P(t, x) with T and sigma ***********************\n")
  call_prices_list, put_prices_list = [], []
  sigma_list = np.linspace(0.01, 1, num = 100)
  T_list = np.linspace(0.1, 5, num = 100)

  sigma_list, T_list = np.meshgrid(sigma_list, T_list)
  row, col = len(sigma_list), len(sigma_list[0])

  for i in range(row):
    call_prices_list.append([])
    put_prices_list.append([])

    for j in range(col):
      C, P = BSM_model(x, t, T_list[i][j], K, r, sigma_list[i][j])
      call_prices_list[i].append(C)
      put_prices_list[i].append(P)

  call_prices_list = np.array(call_prices_list)
  put_prices_list = np.array(put_prices_list)  
  
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(T_list, sigma_list, call_prices_list, cmap=cm.coolwarm)
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.title('C(t, x) vs T and sigma')
  ax.set_xlabel("T") 
  ax.set_ylabel("sigma") 
  ax.set_zlabel("C(t, x")
  plt.show()

  fig = plt.figure()
  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(T_list, sigma_list, put_prices_list, cmap=cm.coolwarm)
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.title('P(t, x) vs T and sigma')
  ax.set_xlabel("T") 
  ax.set_ylabel("sigma") 
  ax.set_zlabel("P(t, x")
  plt.show()


def variation_with_K_and_x(t, T, r, sigma):
  print("*********************** Variation of C(t, x) and P(t, x) with K and x ***********************\n")
  call_prices_list, put_prices_list = [], []
  K_list = np.linspace(0.01, 2, num = 100)
  x_list = np.linspace(0.2, 2, num = 100)

  K_list, x_list = np.meshgrid(K_list, x_list)
  row, col = len(x_list), len(x_list[0])

  for i in range(row):
    call_prices_list.append([])
    put_prices_list.append([])

    for j in range(col):
      C, P = BSM_model(x_list[i][j], t, T, K_list[i][j], r, sigma)
      call_prices_list[i].append(C)
      put_prices_list[i].append(P)

  call_prices_list = np.array(call_prices_list)
  put_prices_list = np.array(put_prices_list)  
  
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(K_list, x_list, call_prices_list, cmap=cm.coolwarm)
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.title('C(t, x) vs K and x')
  ax.set_xlabel("K") 
  ax.set_ylabel("x") 
  ax.set_zlabel("C(t, x)")
  plt.show()

  fig = plt.figure()
  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(K_list, x_list, put_prices_list, cmap=cm.coolwarm)
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.title('P(t, x) vs K and x')
  ax.set_xlabel("K") 
  ax.set_ylabel("x") 
  ax.set_zlabel("P(t, x)")
  plt.show()


def variation_with_T_and_x(t, K, r, sigma):
  print("*********************** Variation of C(t, x) and P(t, x) with T and x ***********************\n")
  call_prices_list, put_prices_list = [], []
  x_list = np.linspace(0.2, 2, num = 100)
  T_list = np.linspace(0.1, 5, num = 100)

  x_list, T_list = np.meshgrid(x_list, T_list)
  row, col = len(x_list), len(x_list[0])

  for i in range(row):
    call_prices_list.append([])
    put_prices_list.append([])

    for j in range(col):
      C, P = BSM_model(x_list[i][j], t, T_list[i][j], K, r, sigma)
      call_prices_list[i].append(C)
      put_prices_list[i].append(P)

  call_prices_list = np.array(call_prices_list)
  put_prices_list = np.array(put_prices_list)  
  
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(T_list, x_list, call_prices_list, cmap=cm.coolwarm)
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.title('C(t, x) vs T and x')
  ax.set_xlabel("T") 
  ax.set_ylabel("x") 
  ax.set_zlabel("C(t, x)")
  plt.show()

  fig = plt.figure()
  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(T_list, x_list, put_prices_list, cmap=cm.coolwarm)
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.title('P(t, x) vs T and x')
  ax.set_xlabel("T") 
  ax.set_ylabel("x") 
  ax.set_zlabel("P(t, x)")
  plt.show()


def variation_with_x_and_r(t, K, T, sigma):
  print("*********************** Variation of C(t, x) and P(t, x) with x and r ***********************\n")
  call_prices_list, put_prices_list = [], []
  x_list = np.linspace(0.2, 2, num = 100)
  r_list = np.linspace(0.01, 1, num = 100)

  x_list, r_list = np.meshgrid(x_list, r_list)
  row, col = len(x_list), len(x_list[0])

  for i in range(row):
    call_prices_list.append([])
    put_prices_list.append([])

    for j in range(col):
      C, P = BSM_model(x_list[i][j], t, T, K, r_list[i][j], sigma)
      call_prices_list[i].append(C)
      put_prices_list[i].append(P)

  call_prices_list = np.array(call_prices_list)
  put_prices_list = np.array(put_prices_list)  
  
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(x_list, r_list, call_prices_list, cmap=cm.coolwarm)
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.title('C(t, x) vs x and r')
  ax.set_xlabel("x") 
  ax.set_ylabel("r") 
  ax.set_zlabel("C(t, x)")
  plt.show()

  fig = plt.figure()
  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(x_list, r_list, put_prices_list, cmap=cm.coolwarm)
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.title('P(t, x) vs x and r')
  ax.set_xlabel("x") 
  ax.set_ylabel("r") 
  ax.set_zlabel("P(t, x)")
  plt.show()


def variation_with_x_and_sigma(t, K, T, r):
  print("*********************** Variation of C(t, x) and P(t, x) with x and sigma ***********************\n")
  call_prices_list, put_prices_list = [], []
  x_list = np.linspace(0.2, 2, num = 100)
  sigma_list = np.linspace(0.01, 1, num = 100)

  x_list, sigma_list = np.meshgrid(x_list, sigma_list)
  row, col = len(x_list), len(x_list[0])

  for i in range(row):
    call_prices_list.append([])
    put_prices_list.append([])

    for j in range(col):
      C, P = BSM_model(x_list[i][j], t, T, K, r, sigma_list[i][j])
      call_prices_list[i].append(C)
      put_prices_list[i].append(P)

  call_prices_list = np.array(call_prices_list)
  put_prices_list = np.array(put_prices_list)  
  
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(x_list, sigma_list, call_prices_list, cmap=cm.coolwarm)
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.title('C(t, x) vs x and sigma')
  ax.set_xlabel("x") 
  ax.set_ylabel("sigma") 
  ax.set_zlabel("C(t, x)")
  plt.show()

  fig = plt.figure()
  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(x_list, sigma_list, put_prices_list, cmap=cm.coolwarm)
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.title('P(t, x) vs x and sigma')
  ax.set_xlabel("x") 
  ax.set_ylabel("sigma") 
  ax.set_zlabel("P(t, x)")
  plt.show()


def main():
  variation_with_T(1, 1, 0.05, 0.6)
  variation_with_K(1, 1, 0.05, 0.6)
  variation_with_r(1, 1, 0.05, 0.6)
  variation_with_sigma(1, 1, 0.05, 0.6)
  variation_with_K_and_r(0.8, 0, 1, 0.6)
  variation_with_K_and_sigma(0.8, 0, 1, 0.05)
  variation_with_r_and_sigma(0.8, 0, 1, 1)
  variation_with_T_and_K(0.8, 0, 0.05, 0.6)
  variation_with_T_and_r(0.8, 0, 1, 0.6)
  variation_with_T_and_sigma(0.8, 0, 1, 0.05)
  variation_with_K_and_x(0, 1, 0.05, 0.6)
  variation_with_T_and_x(0, 1, 0.05, 0.6)
  variation_with_x_and_r(0, 1, 1, 0.6)
  variation_with_x_and_sigma(0, 1, 1, 0.05)


if __name__=="__main__":
  main()