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
  print("**********  Beta for securities in BSE  **********")
  stocks_name_BSE = ['WIPRO.BO', 'BAJAJ-AUTO.BO', 'HDFCBANK.BO', 'HEROMOTOCO.BO', 'TCS.BO',
            'INFY.BO', 'NESTLEIND.BO', 'MARUTI.BO', 'RELIANCE.BO', 'TATAMOTORS.BO']
  beta_BSE = compute_beta(stocks_name_BSE, './180123053_Data/BSE/BSESN.csv', 'BSE Index')

  for i in range(len(beta_BSE)):
    print("{}\t\t=\t\t{}".format(stocks_name_BSE[i], beta_BSE[i]))



  print("\n\n**********  Beta for securities in NSE  **********")
  stocks_name_NSE = ['WIPRO.NS', 'BAJAJ-AUTO.NS', 'HDFCBANK.NS', 'HEROMOTOCO.NS', 'TCS.NS',
            'INFY.NS', 'NESTLEIND.NS', 'MARUTI.NS', 'RELIANCE.NS', 'TATAMOTORS.NS']
  beta_NSE = compute_beta(stocks_name_NSE, './180123053_Data/NSE/NSEI.csv', 'NSE Index')
  
  for i in range(len(beta_NSE)):
    print("{}\t\t=\t\t{}".format(stocks_name_NSE[i], beta_NSE[i]))
    
    

  print("\n\n**********  Beta for securities in non-index using BSE Index  **********")
  stocks_name_non = ['ACC.NS', 'CUMMINSIND.NS', 'EMAMILTD.NS', 'GODREJIND.NS', 'IBULHSGFIN.NS',
            'LUPIN.NS', 'MAHABANK.NS', 'PNB.NS', 'TATACHEM.NS', 'ZYDUSWELL.NS']
  beta_non_BSE = compute_beta(stocks_name_non, './180123053_Data/BSE/BSESN.csv', 'Non-index')
  
  for i in range(len(beta_non_BSE)):
    print("{}\t\t=\t\t{}".format(stocks_name_non[i], beta_non_BSE[i]))
    
    
  

  print("\n\n**********  Beta for securities in non-index using NSE Index  **********")
  stocks_name_non = ['ACC.NS', 'CUMMINSIND.NS', 'EMAMILTD.NS', 'GODREJIND.NS', 'IBULHSGFIN.NS',
            'LUPIN.NS', 'MAHABANK.NS', 'PNB.NS', 'TATACHEM.NS', 'ZYDUSWELL.NS']
  beta_non_NSE = compute_beta(stocks_name_non, './180123053_Data/NSE/NSEI.csv', 'Non-index')
  
  for i in range(len(beta_non_NSE)):
    print("{}\t\t=\t\t{}".format(stocks_name_non[i], beta_non_NSE[i]))


if __name__=="__main__":
  main()