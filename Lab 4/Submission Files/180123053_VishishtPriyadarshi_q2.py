"""
Kindly install these libraries before executing this code:
  1. numpy
  2. matplotlib
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import random

# if using a Jupyter notebook, kindly uncomment the following line:
# %matplotlib inline


def plot_q2(x1, y1, x2, y2, x3, y3, x4, y4, x_axis, y_axis, title):
  plt.plot(x1, y1, color = 'Blue', label = '3 stocks')
  plt.plot(x2, y2, color = 'Green', label = 'Stock 1 and 2')
  plt.plot(x3, y3, color = 'Orange', label = 'Stock 2 and 3')
  plt.plot(x4, y4, color = 'Red', label = 'Stock 1 and 3')
  plt.xlabel(x_axis)
  plt.ylabel(y_axis) 
  plt.title(title)
  plt.grid(True)
  plt.legend()
  plt.show()


def plot_q2_with_scatter(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x_axis, y_axis, title):
  plt.plot(x1, y1, color = 'Blue', linewidth = 5, label = '3 stocks')
  plt.plot(x2, y2, color = 'Green', linewidth = 5, label = 'Stock 1 and 2')
  plt.plot(x3, y3, color = 'Orange', linewidth = 5, label = 'Stock 2 and 3')
  plt.plot(x4, y4, color = 'Red', linewidth = 5, label = 'Stock 1 and 3')
  plt.scatter(x5, y5, color = 'm', marker = '.', label = 'Feasible region')
  plt.xlabel(x_axis)
  plt.ylabel(y_axis) 
  plt.title(title)
  plt.grid(True)
  plt.legend()
  plt.show()


def compute_weights(M, C, mu):
  n = len(M)
  C_inverse = np.linalg.inv(C)
  u = [1 for i in range(n)]

  p = [[1, u @ C_inverse @ np.transpose(M)], [mu, M @ C_inverse @ np.transpose(M)]]
  q = [[u @ C_inverse @ np.transpose(u), 1], [M @ C_inverse @ np.transpose(u), mu]]
  r = [[u @ C_inverse @ np.transpose(u), u @ C_inverse @ np.transpose(M)], [M @ C_inverse @ np.transpose(u), M @ C_inverse @ np.transpose(M)]]

  det_p, det_q, det_r = np.linalg.det(p), np.linalg.det(q), np.linalg.det(r)
  det_p /= det_r
  det_q /= det_r

  w = det_p * (u @ C_inverse) + det_q * (M @ C_inverse)
  
  return w


def minimum_variance_portfolio(M, C):
  u = [1 for i in range(len(M))]

  weight_min_var = u @ np.linalg.inv(C) / (u @ np.linalg.inv(C) @ np.transpose(u))
  mu_min_var = weight_min_var @ np.transpose(M)
  risk_min_var = math.sqrt(weight_min_var @ C @ np.transpose(weight_min_var))

  return risk_min_var, mu_min_var


def get_eqn(x, y):
  slope, intercept = [], []
  for i in range(len(x) - 1):
    x1, x2 = x[i], x[i + 1]
    y1, y2 = y[i], y[i + 1]
    slope.append((y2 - y1)/(x2 - x1))
    intercept.append(y1 - slope[-1]*x1)

  return sum(slope)/len(slope), sum(intercept)/len(intercept)


def main():
  M = [0.1, 0.2, 0.15]
  C = [[0.005, -0.010, 0.004], [-0.010, 0.040, -0.002], [0.004, -0.002, 0.023]]

  returns = np.linspace(0, 0.5, num = 10000)
  risk, actual_returns, weights = [], [], []
  risk_feasible_region, returns_feasible_region = [], []

  for mu in returns:
    w = compute_weights(M, C, mu)
    if w[0] < 0 or w[1] < 0 or w[2] < 0:
      continue
      
    weights.append(w)
    sigma = math.sqrt(w @ C @ np.transpose(w))
    risk.append(sigma)
    actual_returns.append(mu)
  
  for i in range(500):
    w1, w2, w3 = random.randint(100), random.randint(100), random.randint(100)
    normalisation_term = w1 + w2 + w3

    while normalisation_term == 0:
      w1, w2, w3 = random.randint(100), random.randint(100), random.randint(100)
      normalisation_term = w1 + w2 + w3

    w1 /= normalisation_term
    w2 /= normalisation_term
    w3 /= normalisation_term
    w = np.array([w1, w2, w3])

    returns_feasible_region.append(M @ np.transpose(w))
    risk_feasible_region.append(math.sqrt(w @ C @ np.transpose(w)))
    
  risk_min_var, mu_min_var = minimum_variance_portfolio(M, C)
  returns_plot1, risk_plot1, returns_plot2, risk_plot2 = [], [], [], []

  for i in range(len(actual_returns)):
    if actual_returns[i] >= mu_min_var: 
      returns_plot1.append(actual_returns[i])
      risk_plot1.append(risk[i])
    else:
      returns_plot2.append(actual_returns[i])
      risk_plot2.append(risk[i])

  plt.plot(risk_plot1, returns_plot1, color = 'green', linewidth = 5, label = 'Efficient frontier')
  plt.plot(risk_plot2, returns_plot2, color = 'blue', linewidth = 5)
  plt.scatter(risk_feasible_region, returns_feasible_region, color = 'm', linewidth = 0.1, label = 'Feasible region')
  plt.scatter(risk_min_var, mu_min_var, color = 'orange', linewidth = 5, label = 'Minimum Variance Point')
  plt.xlabel("Risk (sigma)")
  plt.ylabel("Returns") 
  plt.title("Minimum Variance Curve")
  plt.grid(True)
  plt.legend()
  plt.show()

  M_1 = [0.1, 0.2]
  C_1 = [[0.005, -0.010], [-0.010, 0.040]]
  risk_1, actual_returns_1, weights_1 = [], [], []
  
  for mu in returns:
    w = compute_weights(M_1, C_1, mu)
    if w[0] < 0 or w[1] < 0:
      continue
    weights_1.append(w)
    sigma = math.sqrt(w @ C_1 @ np.transpose(w))
    risk_1.append(sigma)
    actual_returns_1.append(mu)
  

  M_2 = [0.2, 0.15]
  C_2 = [[0.040, -0.002], [-0.002, 0.023]]
  risk_2, actual_returns_2, weights_2 = [], [], []

  for mu in returns:
    w = compute_weights(M_2, C_2, mu)
    if w[0] < 0 or w[1] < 0:
      continue
    weights_2.append(w)
    sigma = math.sqrt(w @ C_2 @ np.transpose(w))
    risk_2.append(sigma)
    actual_returns_2.append(mu)

  
  M_3 = [0.1, 0.15]
  C_3 = [[0.005, 0.004], [0.004, 0.023]]
  risk_3, actual_returns_3, weights_3 = [], [], []

  for mu in returns:
    w = compute_weights(M_3, C_3, mu)
    if w[0] < 0 or w[1] < 0:
      continue
    weights_3.append(w)
    sigma = math.sqrt(w @ C_3 @ np.transpose(w))
    risk_3.append(sigma)
    actual_returns_3.append(mu)
  

  # ========================= plot for the weights =========================
  plot_q2(risk, actual_returns, risk_1, actual_returns_1, risk_2, actual_returns_2, risk_3, actual_returns_3, "Sigma (risk)", "Returns", "Minimum Variance Curve - No short sales")
  plot_q2_with_scatter(risk, actual_returns, risk_1, actual_returns_1, risk_2, actual_returns_2, risk_3, actual_returns_3, risk_feasible_region, returns_feasible_region, "Sigma (risk)", "Returns", "Minimum Variance Curve (with feasible region) - No short sales")

  weights.clear()
  risk.clear()
  for mu in returns:
    w = compute_weights(M, C, mu)  
    weights.append(w)
    sigma = math.sqrt(w @ C @ np.transpose(w))
    risk.append(sigma)
    
  w_01, w_02, w_03 = np.array([i[0] for i in weights]), np.array([i[1] for i in weights]), np.array([i[2] for i in weights])
  x = np.linspace(-5, 5, 1000)
  y = [0 for i in range(len(x))]

  m, c = get_eqn(w_01, w_02)
  print("Eqn of line w1 vs w2 is:")
  print("w2 = {:.2f} w1 + {:.2f}".format(m, c))
  plt.axis([-0.5, 1.5, -0.5, 1.5])
  plt.plot(w_01, w_02, color = 'Blue', label = 'w1 vs w2')
  plt.plot(w_01, 1 - w_01, color = 'Green', label = 'w1 + w2 = 1')
  plt.plot(x, y, color = 'red', label = 'w2 = 0')
  plt.plot(y, x, color = 'yellow', label = 'w1 = 0')
  plt.xlabel("weights_1 (w1)")
  plt.ylabel("weights_2 (w2)") 
  plt.title("weights corresponding to min variance curve (w1 vs w2)")
  plt.grid(True)
  plt.legend()
  plt.show()

  
  m, c = get_eqn(w_02, w_03)
  print("Eqn of line w2 vs w3 is:")
  print("w3 = {:.2f} w2 + {:.2f}".format(m, c))
  plt.axis([-0.5, 1.5, -0.5, 1.5])
  plt.plot(w_02, w_03, color = 'Blue', label = 'w2 vs w3')
  plt.plot(w_02, 1 - w_02, color = 'Green', label = 'w2 + w3 = 1')
  plt.plot(x, y, color = 'red', label = 'w3 = 0')
  plt.plot(y, x, color = 'yellow', label = 'w2 = 0')
  plt.xlabel("weights_2 (w2)")
  plt.ylabel("weights_3 (w3)") 
  plt.title("weights corresponding to min variance curve (w2 vs w3)")
  plt.grid(True)
  plt.legend()
  plt.show()

  
  m, c = get_eqn(w_01, w_03)
  print("Eqn of line w1 vs w3 is:")
  print("w3 = {:.2f} w1 + {:.2f}".format(m, c))
  plt.axis([-0.5, 1.5, -0.5, 1.5])
  plt.plot(w_01, w_03, color = 'Blue', label = 'w1 vs w3')
  plt.plot(w_03, 1 - w_03, color = 'Green', label = 'w1 + w3 = 1')
  plt.plot(x, y, color = 'red', label = 'w3 = 0')
  plt.plot(y, x, color = 'yellow', label = 'w1 = 0')
  plt.xlabel("weights_1 (w1)")
  plt.ylabel("weights_3 (w3)") 
  plt.title("weights corresponding to min variance curve (w1 vs w3)")
  plt.grid(True)
  plt.legend()
  plt.show()


if __name__=="__main__":
  main()