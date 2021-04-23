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
from pandas import to_datetime

# if using a Jupyter notebook, kindly uncomment the following line:
# %matplotlib inline


def plot_stock_prices(df, stocks_name, stocks_type):
  interval = ['daily', 'weekly', 'monthly']
  df_initial = df.copy()

  for sname in stocks_name:
    df = df_initial.copy()

    for intvl in interval:
      if intvl == 'weekly':
        df['Day'] = (to_datetime(df['Date'])).dt.day_name()
        df = df.loc[df['Day'] == 'Monday']
        del df['Day']
      elif intvl == 'monthly':
        df = df.groupby(pd.DatetimeIndex(df['Date']).to_period('M')).nth(0)

      x = df['Date'].to_list()
      y = df[sname].to_list()
      
      plt.rcParams["figure.figsize"] = (20, 5)
      if intvl == 'daily':
        plt.subplot(1, 3, 1)
      elif intvl == 'weekly':
        plt.subplot(1, 3, 2)
      else:
        plt.subplot(1, 3, 3)

      plt.plot(x, y)
      plt.xticks(np.arange(0, len(x), int(len(x)/4)), df['Date'][0:len(x):int(len(x)/4)])
      plt.title('Plot for Stock prices for {} on {} basis'.format(sname, intvl))
      plt.xlabel('Time')
      plt.ylabel('Price')
      plt.grid(True)

      if intvl == 'monthly':
        # plt.savefig('./Plots/' + stocks_type + '/' + sname + '.png')
        plt.show()


def main():
  stocks_type = ['BSE', 'NSE']
  stocks_name = []

  for fname in stocks_type:
    filename = ""
    if fname == 'BSE':
      stocks_name = ['WIPRO.BO', 'BAJAJ-AUTO.BO', 'HDFCBANK.BO', 'HEROMOTOCO.BO', 'TCS.BO',
            'INFY.BO', 'NESTLEIND.BO', 'MARUTI.BO', 'RELIANCE.BO', 'TATAMOTORS.BO', 'BSE Index']
      filename = "bsedata1.csv"
    else:
      stocks_name = ['WIPRO.NS', 'BAJAJ-AUTO.NS', 'HDFCBANK.NS', 'HEROMOTOCO.NS', 'TCS.NS',
            'INFY.NS', 'NESTLEIND.NS', 'MARUTI.NS', 'RELIANCE.NS', 'TATAMOTORS.NS', 'NSE Index']
      filename = "nsedata1.csv"
    
    df = pd.read_csv(filename)
    plot_stock_prices(df, stocks_name, fname)
      

if __name__=="__main__":
  main()