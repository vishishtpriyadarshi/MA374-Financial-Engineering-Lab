"""
Kindly install these libraries before executing this code:
  1. numpy
  2. tabulate
  3. pandas
"""

import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.stats import norm

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
  
  return volatility


def BSM_model(x, t, T, K, r, sigma):
  if t == T:
    return max(0, x - K), max(0, K - x)

  d1 = ( math.log(x/K) + (r + 0.5 * sigma * sigma) * (T - t) ) / ( sigma * math.sqrt(T - t) )
  d2 = ( math.log(x/K) + (r - 0.5 * sigma * sigma) * (T - t) ) / ( sigma * math.sqrt(T - t) )

  call_price = x * norm.cdf(d1) - K * math.exp( -r * (T - t) ) * norm.cdf(d2)
  put_price = K * math.exp( -r * (T - t) ) * norm.cdf(-d2) - x * norm.cdf(-d1)

  return call_price, put_price


def price_computation(stocks_type):
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
  df = df.interpolate(method ='linear', limit_direction ='forward')
  r = 0.05
  t = 0
  T = 6/12
  sigma_list = []
  time_period = range(1, 61)

  for delta_t in range(1, 61):
    sigma_list.append(get_historical_volatility(stocks_type, delta_t))
    
  for idx1 in range(len(stocks_name)):
    print("***********************  For stocks - {} ***********************".format(stocks_name[idx1]))
    plt.rcParams["figure.figsize"] = (20, 10)
    S0 = df.iloc[len(df) - 1][stocks_name[idx1]]
    call_prices, put_prices = np.zeros((11, 60)), np.zeros((11, 60))
    historical_volatility = []
    
    for idx2 in range(60):
      sigma = sigma_list[idx2][idx1]
      historical_volatility.append(sigma)
      A = [round(0.1 * i, 2) for i in range(5, 16)]

      for idx3 in range(len(A)):
        K = A[idx3] * S0
        call, put = BSM_model(S0, t, T, K, r, sigma)
        call_prices[idx3][idx2] = call
        put_prices[idx3][idx2] = put
    
    for i in range(len(A)):
      ax = plt.subplot(2, 3, (i % 6) + 1)
      plt.plot(time_period, call_prices[i])
      plt.xlabel("Length of time period (in months)")
      plt.ylabel("European Call Option Price")
      ax.set_title("Call Option for {} with K = {}*S0".format(stocks_name[idx1], A[i]))
      if i == 5:
        # plt.savefig('./Data/' + stocks_name[idx1] + '_Call_1.jpg')
        plt.show() 
    # plt.savefig('./Data/' + stocks_name[idx1] + '_Call_2.jpg')
    plt.show()

    for i in range(len(A)):
      ax = plt.subplot(2, 3, (i % 6) + 1)
      plt.plot(time_period, put_prices[i])
      plt.xlabel("Length of time period (in months)")
      plt.ylabel("European Put Option Price")
      ax.set_title("Put Option for {} with K = {}*S0".format(stocks_name[idx1], A[i]))
      if i == 5:
        # plt.savefig('./Data/' + stocks_name[idx1] + '_Put_1.jpg')
        plt.show()
    # plt.savefig('./Data/' + stocks_name[idx1] + '_Put_2.jpg')
    plt.show()

    plt.plot(time_period, historical_volatility)
    plt.xlabel("Length of time period (in months)")
    plt.ylabel("Volatility")
    plt.title("Historical Volatility vs time period for {}".format(stocks_name[idx1]))
    # plt.savefig('./Data/' + stocks_name[idx1] + '_volatility.jpg')
    plt.show()


def main():
  price_computation('BSE')
  price_computation('NSE')
    

if __name__=="__main__":
  main()