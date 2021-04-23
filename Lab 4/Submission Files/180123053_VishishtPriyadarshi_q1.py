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


def compute_weights(M, C, mu):
  C_inverse = np.linalg.inv(C)
  u = [1 for i in range(len(M))]

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


def main():
  # ==================  sub-part (a) ==================
  print("==================  sub-part (a) ==================\n")

  M = [0.1, 0.2, 0.15]
  C = [[0.005, -0.010, 0.004], [-0.010, 0.040, -0.002], [0.004, -0.002, 0.023]]

  returns = np.linspace(0, 0.5, num = 10000)
  risk = []

  weights_10, return_10, risk_10 = [], [], []
  weights_15, return_15 = [], []
  weights_18, risk_18 = [], []

  ct = 0
  for mu in returns:
    w = compute_weights(M, C, mu)
    sigma = math.sqrt(w @ C @ np.transpose(w))
    risk.append(sigma)

    ct += 1
    if ct % 1000 == 0:
      weights_10.append(w)
      return_10.append(mu)
      risk_10.append(sigma*sigma)

    if abs(sigma - 0.15) < math.pow(10, -4.5):
      weights_15.append(w)
      return_15.append(mu)

  risk_min_var, mu_min_var = minimum_variance_portfolio(M, C)
  returns_plot1, risk_plot1, returns_plot2, risk_plot2 = [], [], [], []

  for i in range(len(returns)):
    if returns[i] >= mu_min_var: 
      returns_plot1.append(returns[i])
      risk_plot1.append(risk[i])
    else:
      returns_plot2.append(returns[i])
      risk_plot2.append(risk[i])

  plt.plot(risk_plot1, returns_plot1, color = 'yellow', label = 'Efficient frontier')
  plt.plot(risk_plot2, returns_plot2, color = 'blue')
  plt.xlabel("Risk (sigma)")
  plt.ylabel("Returns") 
  plt.title("Minimum variance line along with Markowitz Efficient Frontier")
  plt.plot(risk_min_var, mu_min_var, color = 'green', marker = 'o')
  plt.annotate('Minimum Variance Portfolio (' + str(round(risk_min_var, 2)) + ', ' + str(round(mu_min_var, 2)) + ')', 
             xy=(risk_min_var, mu_min_var), xytext=(risk_min_var + 0.05, mu_min_var))
  plt.legend()
  plt.grid(True)
  plt.show()
  

  # ==================  sub-part (b) ==================
  print("\n\n==================  sub-part (b) ==================\n")
  print("Index\tweights\t\t\t\t\trisk\t\t\treturn\n")
  for i in range(10):
    print("{}.\t{}\t{}\t{}".format(i + 1, weights_10[i], return_10[i], risk_10[i]))


  # ==================  sub-part (c) ==================
  print("\n\n==================  sub-part (c) ==================\n")
  min_return, max_return = return_15[0], return_15[1]
  min_return_weights, max_return_weights = weights_15[0], weights_15[1]

  if min_return > max_return:
    min_return, max_return = max_return, min_return
    min_return_weights, max_return_weights = max_return_weights, min_return_weights

  print("Minimum return \t= {}".format(min_return))
  print("weights\t\t= {}".format(min_return_weights))
  
  print("\nMaximum return \t= {}".format(max_return))
  print("weights\t\t= {}".format(max_return_weights))


  # ==================  sub-part (d) ==================
  print("\n\n==================  sub-part (d) ==================\n")
  given_return = 0.18
  w = compute_weights(M, C, given_return)
  minimum_risk = math.sqrt(w @ C @ np.transpose(w))

  print("Minimum risk for 18% return \t= ", minimum_risk * 100, " %")
  print("Weights\t\t\t\t= ", w)


  # ==================  sub-part (e) ==================
  print("\n\n==================  sub-part (e) ==================\n")
  mu_rf = 0.1
  u = np.array([1, 1, 1])

  market_portfolio_weights = (M - mu_rf * u) @ np.linalg.inv(C) / ((M - mu_rf * u) @ np.linalg.inv(C) @ np.transpose(u) )
  mu_market = market_portfolio_weights @ np.transpose(M)
  risk_market = math.sqrt(market_portfolio_weights @ C @ np.transpose(market_portfolio_weights))

  print("Market Portfolio Weights \t= ", market_portfolio_weights)
  print("Return \t\t\t\t= ", mu_market)
  print("Risk \t\t\t\t= ", risk_market * 100 , " %")

  returns_cml = []
  risk_cml = np.linspace(0, 1, num = 10000)
  for i in risk_cml:
    returns_cml.append(mu_rf + (mu_market - mu_rf) * i / risk_market)

  slope, intercept = (mu_market - mu_rf) / risk_market, mu_rf
  
  print("\nEquation of CML is:")
  print("y = {:.2f} x + {:.2f}\n".format(slope, intercept))
  

  plt.scatter(risk_market, mu_market, color = 'orange', linewidth = 3, label = 'Market portfolio')
  plt.plot(risk, returns, color = 'blue', label = 'Minimum variance curve')
  plt.plot(risk_cml, returns_cml, color = 'green', label = 'CML')
  plt.xlabel("Risk (sigma)")
  plt.ylabel("Returns") 
  plt.title("Capital Market Line with Minimum variance curve")
  plt.grid(True)
  plt.legend()
  plt.show()


  # ==================  sub-part (f) ==================
  print("\n\n==================  sub-part (f) ==================\n")
  sigma = 0.1
  mu_curr = (mu_market - mu_rf) * sigma / risk_market + mu_rf
  weight_rf = (mu_curr - mu_market) / (mu_rf - mu_market)
  weights_risk = (1 - weight_rf) * market_portfolio_weights

  print("Risk \t\t\t=", sigma * 100, " %")
  print("Risk-free weights \t=", weight_rf)
  print("Risky Weights \t\t=", weights_risk)
  print("Returns\t\t\t=", mu_curr)

  sigma = 0.25
  mu_curr = (mu_market - mu_rf) * sigma / risk_market + mu_rf
  weight_rf = (mu_curr - mu_market) / (mu_rf - mu_market)
  weights_risk = (1 - weight_rf) * market_portfolio_weights

  print("\n\nRisk \t\t\t=", sigma * 100, " %")
  print("Risk-free weights \t=", weight_rf)
  print("Risky Weights \t\t=", weights_risk)
  print("Returns\t\t\t=", mu_curr)


if __name__=="__main__":
  main()