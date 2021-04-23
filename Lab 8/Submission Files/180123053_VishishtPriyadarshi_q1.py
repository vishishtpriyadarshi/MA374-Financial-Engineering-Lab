"""
Kindly install these libraries before executing this code:
  1. numpy
  2. tabulate
  3. pandas
  4. matplotlib
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from tabulate import tabulate

# if using a Jupyter notebook, kindly uncomment the following line:
# %matplotlib inline

def get_historical_volatility(stocks_type, time_period):
  filename, stocks_name = '', []
  if stocks_type == 'BSE':
    stocks_name = ['WIPRO.BO', 'BAJAJ-AUTO.BO', 'HDFCBANK.BO', 'HEROMOTOCO.BO', 'TCS.BO',
            'INFY.BO', 'NESTLEIND.BO', 'MARUTI.BO', 'RELIANCE.BO', 'TATAMOTORS.BO', 'BSE Index']
    filename = './bsedata1.csv'
  else:
    stocks_name = ['WIPRO.NS', 'BAJAJ-AUTO.NS', 'HDFCBANK.NS', 'HEROMOTOCO.NS', 'TCS.NS',
            'INFY.NS', 'NESTLEIND.NS', 'MARUTI.NS', 'RELIANCE.NS', 'TATAMOTORS.NS', 'NSE Index']
    filename = './nsedata1.csv'
  
  df = pd.read_csv(filename)
  df_monthly = df.groupby(pd.DatetimeIndex(df.Date).to_period('M')).nth(0)

  start_idx = 60 - time_period
  df_reduced = df_monthly.iloc[start_idx :]
  df_reduced.reset_index(inplace = True, drop = True) 
  idx_list = df.index[df['Date'] >= df_reduced.iloc[0]['Date']].tolist()
  df_reduced = df.iloc[idx_list[0] :]

  data = df_reduced.set_index('Date')
  data = data.pct_change()

  volatility = []
  for sname in stocks_name:
    returns = data[sname]
    x = returns.to_list()
    mean = np.nanmean(np.array(x))
    std = np.nanstd(np.array(x))
    
    volatility.append(std * math.sqrt(252))
  
  table = []
  for i in range(len(volatility)):
    table.append([i + 1, stocks_name[i], volatility[i]])
  
  print(tabulate(table, headers = ['SI No', 'Stocks Name', 'Historical Volatility']))


def main():
  print("**********  Historical Volatility of last month for BSE  **********")
  get_historical_volatility('BSE', 1)

  print("\n\n**********  Historical Volatility of last month for NSE  **********")
  get_historical_volatility('NSE', 1)
    

if __name__=="__main__":
  main()