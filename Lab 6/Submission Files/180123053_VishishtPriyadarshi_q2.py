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
import scipy.stats as stats

# if using a Jupyter notebook, kindly uncomment the following line:
# %matplotlib inline


def plot_returns(df, stocks_name, stocks_type):
  interval = ['daily', 'weekly', 'monthly']
  df_initial = df.copy()
  plt.rcParams["figure.figsize"] = (20, 5)

  for sname in stocks_name:
    df = df_initial.copy()

    for intvl in interval:
      if intvl == 'weekly':
        df['Day'] = (to_datetime(df['Date'])).dt.day_name()
        df = df.loc[df['Day'] == 'Monday']
        del df['Day']
      elif intvl == 'monthly':
        df = df.groupby(pd.DatetimeIndex(df['Date']).to_period('M')).nth(0)

      data = df.set_index('Date')
      data = data.pct_change()
      returns = data[sname]

      x = returns.to_list()
      mean = np.nanmean(np.array(x))
      std = np.nanstd(np.array(x))
      x = [(i - mean)/std for i in x]

      if intvl == 'daily':
        plt.subplot(1, 3, 1)
      elif intvl == 'weekly':
        plt.subplot(1, 3, 2)
      else:
        plt.subplot(1, 3, 3)

      n_bins = 40
      plt.hist(x, n_bins, density = True, edgecolor = 'black', linewidth = 0.4, color = 'yellow', label = 'Normalized returns')
      mu = 0
      variance = 1
      sigma = math.sqrt(variance)
      x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
      
      plt.plot(x, stats.norm.pdf(x, mu, sigma), color = 'green', label = 'density function, N(0, 1)')
      plt.xlabel('Returns')
      plt.ylabel('Normalised Frequency')
      plt.title('Normalized returns with N(0, 1) for {} on {} basis'.format(sname, intvl))
      plt.legend()

      if intvl == 'monthly':
        # plt.savefig('./Histograms/' + stocks_type + '/' + sname + '.png')   
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
    plot_returns(df, stocks_name, fname)


if __name__=="__main__":
  main()