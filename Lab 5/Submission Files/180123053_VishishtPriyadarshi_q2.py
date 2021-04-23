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

  return mu_market, risk_market



def execute_model(stocks_name, type, mu_market_index, risk_market_index, beta):
  daily_returns = []
  mu_rf = 0.05

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
  
  print("\n\nStocks Name\t\t\tActual Return\t\t\tExpected Return\n")
  for i in range(len(M)):
    print("{}\t\t\t{}\t\t{}".format(stocks_name[i], M[i], beta[i] * (mu_market_index - mu_rf) + mu_rf))


def compute_beta(stocks_name, main_filename, index_type):
  df = pd.read_csv(main_filename)
  df.set_index('Date', inplace=True)
  daily_returns = (df['Open'] - df['Close'])/df['Open']

  daily_returns_stocks = []
    
  for i in range(len(stocks_name)):
    if index_type == 'Non-index':
      filename = './180123053_Data/Non-index stocks/' + stocks_name[i] + '.csv'
    else:
      filename = './180123053_Data/' + index_type[:3] + '/' + stocks_name[i] + '.csv'
    df_stocks = pd.read_csv(filename)
    df_stocks.set_index('Date', inplace=True)

    daily_returns_stocks.append((df_stocks['Open'] - df_stocks['Close'])/df_stocks['Open'])
    

  beta_values = []
  for i in range(len(stocks_name)):
    df_combined = pd.concat([daily_returns_stocks[i], daily_returns], axis = 1, keys = [stocks_name[i], index_type])
    C = df_combined.cov()

    beta = C[index_type][stocks_name[i]]/C[index_type][index_type]
    beta_values.append(beta)

  return beta_values


def main():
  print("**********  Inference about stocks taken from BSE  **********")
  stocks_name_BSE = ['WIPRO.BO', 'BAJAJ-AUTO.BO', 'HDFCBANK.BO', 'HEROMOTOCO.BO', 'TCS.BO',
            'INFY.BO', 'NESTLEIND.BO', 'MARUTI.BO', 'RELIANCE.BO', 'TATAMOTORS.BO']
  beta_BSE = compute_beta(stocks_name_BSE, './180123053_Data/BSE/BSESN.csv', 'BSE Index')
  mu_market_BSE, risk_market_BSE = find_market_portfolio('./180123053_Data/BSE/BSESN.csv')
  execute_model(stocks_name_BSE, 'BSE', mu_market_BSE, risk_market_BSE, beta_BSE)



  print("\n\n**********  Inference about stocks taken from NSE  **********")
  stocks_name_NSE = ['WIPRO.NS', 'BAJAJ-AUTO.NS', 'HDFCBANK.NS', 'HEROMOTOCO.NS', 'TCS.NS',
            'INFY.NS', 'NESTLEIND.NS', 'MARUTI.NS', 'RELIANCE.NS', 'TATAMOTORS.NS']
  beta_NSE = compute_beta(stocks_name_NSE, './180123053_Data/NSE/NSEI.csv', 'NSE Index')
  mu_market_NSE, risk_market_NSE = find_market_portfolio('./180123053_Data/NSE/NSEI.csv')
  execute_model(stocks_name_NSE, 'NSE', mu_market_NSE, risk_market_NSE, beta_NSE) 
    
    
    
  print("\n\n**********  Inference about stocks not taken from any index  with index taken from BSE values**********")
  stocks_name_non = ['ACC.NS', 'CUMMINSIND.NS', 'EMAMILTD.NS', 'GODREJIND.NS', 'IBULHSGFIN.NS',
            'LUPIN.NS', 'MAHABANK.NS', 'PNB.NS', 'TATACHEM.NS', 'ZYDUSWELL.NS']
  beta_non_index_BSE = compute_beta(stocks_name_non, './180123053_Data/BSE/BSESN.csv', 'Non-index')
  execute_model(stocks_name_non, 'Non-index stocks', mu_market_BSE, risk_market_BSE, beta_non_index_BSE) 


  print("\n\n**********  Inference about stocks not taken from any index  with index taken from NSE values**********")
  beta_non_index_NSE = compute_beta(stocks_name_non, './180123053_Data/NSE/NSEI.csv', 'Non-index')
  execute_model(stocks_name_non, 'Non-index stocks', mu_market_NSE, risk_market_NSE, beta_non_index_NSE) 


if __name__=="__main__":
  main()