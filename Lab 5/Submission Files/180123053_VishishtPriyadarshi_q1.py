"""
Kindly install these libraries before executing this code:
  1. numpy
  2. matplotlib
  3. pandas
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# if using a Jupyter notebook, kindly uncomment the following line:
# %matplotlib inline


def find_market_portfolio(filename):
  df = pd.read_csv(filename)
  df.set_index('Date', inplace=True)
  daily_returns = (df['Open'] - df['Close'])/df['Open']
  daily_returns = np.array(daily_returns)

  df = pd.DataFrame(np.transpose(daily_returns))
  M, sigma = np.mean(df, axis = 0) * len(df) / 5, df.std()
  
  mu_market = M[0]
  risk_market = sigma[0]
  
  print("Market return \t=", mu_market)
  print("Market risk \t=", risk_market*100, "%")

  return mu_market, risk_market


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


def construct_efficient_frontier(M, C, mu_rf):
  returns = np.linspace(-2, 5, num = 2000)
  u = np.array([1 for i in range(len(M))])
  risk = []

  for mu in returns:
    w = compute_weights(M, C, mu)
    sigma = math.sqrt(w @ C @ np.transpose(w))
    risk.append(sigma)
  
  weight_min_var = u @ np.linalg.inv(C) / (u @ np.linalg.inv(C) @ np.transpose(u))
  mu_min_var = weight_min_var @ np.transpose(M)
  risk_min_var = math.sqrt(weight_min_var @ C @ np.transpose(weight_min_var))

  returns_plot1, risk_plot1, returns_plot2, risk_plot2 = [], [], [], []
  for i in range(len(returns)):
    if returns[i] >= mu_min_var: 
      returns_plot1.append(returns[i])
      risk_plot1.append(risk[i])
    else:
      returns_plot2.append(returns[i])
      risk_plot2.append(risk[i])


  # ==================  Market Portfolio ==================
  market_portfolio_weights = (M - mu_rf * u) @ np.linalg.inv(C) / ((M - mu_rf * u) @ np.linalg.inv(C) @ np.transpose(u) )
  mu_market = market_portfolio_weights @ np.transpose(M)
  risk_market = math.sqrt(market_portfolio_weights @ C @ np.transpose(market_portfolio_weights))

  plt.plot(risk_plot1, returns_plot1, color = 'yellow', label = 'Efficient frontier')
  plt.plot(risk_plot2, returns_plot2, color = 'blue')
  plt.xlabel("Risk (sigma)")
  plt.ylabel("Returns") 
  plt.title("Minimum Variance Curve & Efficient Frontier")
  plt.plot(risk_market, mu_market, color = 'green', marker = 'o')
  plt.annotate('Market Portfolio (' + str(round(risk_market, 4)) + ', ' + str(round(mu_market, 4)) + ')', 
             xy=(risk_market, mu_market), xytext=(0.012, 0.8))
  plt.plot(risk_min_var, mu_min_var, color = 'green', marker = 'o')
  plt.annotate('Minimum Variance Portfolio (' + str(round(risk_min_var, 4)) + ', ' + str(round(mu_min_var, 4)) + ')', 
             xy=(risk_min_var, mu_min_var), xytext=(risk_min_var, -0.6))
  plt.legend()
  plt.grid(True)
  plt.show()

  print("Market Portfolio Weights \t= ", market_portfolio_weights)
  print("Return \t\t\t\t= ", mu_market)
  print("Risk \t\t\t\t= ", risk_market * 100, " %")

  return mu_market, risk_market


def plot_CML(M, C, mu_rf, mu_market, risk_market):
  returns = np.linspace(-2, 5, num = 2000)
  u = np.array([1 for i in range(len(M))])
  risk = []

  for mu in returns:
    w = compute_weights(M, C, mu)
    sigma = math.sqrt(w @ C @ np.transpose(w))
    risk.append(sigma)

  returns_cml = []
  risk_cml = np.linspace(0, 0.25, num = 2000)
  for i in risk_cml:
    returns_cml.append(mu_rf + (mu_market - mu_rf) * i / risk_market)


  slope, intercept = (mu_market - mu_rf) / risk_market, mu_rf
  print("\nEquation of CML is:")
  print("y = {:.4f} x + {:.4f}\n".format(slope, intercept))

  plt.plot(risk_market, mu_market, color = 'green', marker = 'o')
  plt.annotate('Market Portfolio (' + str(round(risk_market, 4)) + ', ' + str(round(mu_market, 4)) + ')', 
             xy=(risk_market, mu_market), xytext=(0.012, 0.8))
  plt.plot(risk, returns, label = 'Minimum Variance Line')
  plt.plot(risk_cml, returns_cml, label = 'CML')
  plt.title("Capital Market Line with Minimum Variance Line")
  plt.xlabel("Risk (sigma)")
  plt.ylabel("Returns")
  plt.grid(True)
  plt.legend()
  plt.show()

  plt.plot(risk_cml, returns_cml)
  plt.title("Capital Market Line")
  plt.xlabel("Risk (sigma)")
  plt.ylabel("Returns")
  plt.grid(True)
  plt.show()


def plot_SML(M, C, mu_rf, mu_market, risk_market):
  beta_k = np.linspace(-1, 1, 2000)
  mu_k = mu_rf + (mu_market - mu_rf) * beta_k
  plt.plot(beta_k, mu_k)
  
  print("Eqn of Security Market Line is:")
  print("mu = {:.2f} beta + {:.2f}".format(mu_market - mu_rf, mu_rf))

  plt.title('Security Market Line for all the 10 assets')
  plt.xlabel("Beta")
  plt.ylabel("Mean Return")
  plt.grid(True)
  plt.show()


def execute_model(stocks_name, type, mu_market_index, risk_market_index):
  daily_returns = []

  for i in range(len(stocks_name)):
    filename = './180123053_Data/' + type + '/' + stocks_name[i] + '.csv'
    df = pd.read_csv(filename)
    df.set_index('Date', inplace=True)

    df = df.pct_change()
    daily_returns.append(df['Open'])

  daily_returns = np.array(daily_returns)
  df = pd.DataFrame(np.transpose(daily_returns), columns = stocks_name)
  M = np.mean(df, axis = 0) * len(df) / 5
  C = df.cov()
  
  mu_market, risk_market = construct_efficient_frontier(M, C, 0.05)
  plot_CML(M, C, 0.05, mu_market, risk_market)

  if type == 'BSE' or type == 'NSE':
    plot_SML(M, C, 0.05, mu_market_index, risk_market_index)
  else:
    plot_SML(M, C, 0.05, mu_market, risk_market)


def main():
  # =====================  sub-part (i)  =====================
  print("********************  Market portfolio for BSE using Index  ********************")
  mu_market_BSE, risk_market_BSE = find_market_portfolio('./180123053_Data/BSE/BSESN.csv')



  # =====================  sub-part (ii)  =====================
  print("\n\n********************  Market portfolio for NSE using Index  ********************")
  mu_market_NSE, risk_market_NSE = find_market_portfolio('./180123053_Data/NSE/NSEI.csv')
  


  # =====================  sub-part (iii)  =====================
  print("\n\n********************  10 stocks from the BSE Index  ********************")

  stocks_name = ['WIPRO.BO', 'BAJAJ-AUTO.BO', 'HDFCBANK.BO', 'HEROMOTOCO.BO', 'TCS.BO',
            'INFY.BO', 'NESTLEIND.BO', 'MARUTI.BO', 'RELIANCE.BO', 'TATAMOTORS.BO']
  execute_model(stocks_name, 'BSE', mu_market_BSE, risk_market_BSE)
  


  # =====================  sub-part (iv)  =====================
  print("\n\n********************  10 stocks from the NSE Index  ********************")

  stocks_name = ['WIPRO.NS', 'BAJAJ-AUTO.NS', 'HDFCBANK.NS', 'HEROMOTOCO.NS', 'TCS.NS',
            'INFY.NS', 'NESTLEIND.NS', 'MARUTI.NS', 'RELIANCE.NS', 'TATAMOTORS.NS']
  execute_model(stocks_name, 'NSE', mu_market_NSE, risk_market_NSE)



  # =====================  sub-part (v)  =====================
  print("\n\n********************  10 stocks not from any Index  ********************")

  stocks_name = ['ACC.NS', 'CUMMINSIND.NS', 'EMAMILTD.NS', 'GODREJIND.NS', 'IBULHSGFIN.NS',
            'LUPIN.NS', 'MAHABANK.NS', 'PNB.NS', 'TATACHEM.NS', 'ZYDUSWELL.NS']
  execute_model(stocks_name, 'Non-index stocks', -1, -1)


if __name__=="__main__":
  main()