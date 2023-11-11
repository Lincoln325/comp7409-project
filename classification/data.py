import numpy as np
import pandas as pd
import yfinance as yf
from hampel import hampel

from utlis import standardize

history = "10y"

short_lookback_period = 15
mid_lookback_period = 60
long_lookback_period = 120
classes = ["Bull","Bear"]
Label_Bull = classes.index("Bull")
Label_Bear = classes.index("Bear")

train_short_x, train_short_y = [],[]
train_mid_x,train_mid_y = [], []
train_long_x,train_long_y = [], []

test_short_x, test_short_y = [],[]
test_mid_x,test_mid_y = [], []
test_long_x,test_long_y = [], []

with open('ticker.txt', 'r') as file:
    # Read each line of the file into a list
    symbols = file.readlines()

# Remove any newline characters from each list item
symbols = [item.strip() for item in symbols]

# Print the list to check the result
print(symbols)



def download_data(symbol, history):
  #download data
  all_day_k = yf.Ticker(symbol).history(period=history, interval="1d")

    # Remove dividend and stock splits column
  all_day_k = all_day_k.drop(columns=["Dividends","Stock Splits"])

    # Remove latest row as it may be imcompleted
  all_day_k = all_day_k[:-1]

  return all_day_k


#Function that use hampel to remove outliers
def remove_outliers(df):
  filtered_df = df.apply(lambda x: hampel(x).filtered_data, axis=0)
  return filtered_df


#split the data into training and testing
def spliting_data(all_day_k, training_data_ratio):
  #get number of row in all_day_k
  num_row = len(all_day_k)
  #get the index of the row to split
  split_index = int(num_row*training_data_ratio)
  #split the data
  training_data = all_day_k[:split_index]
  testing_data = all_day_k[split_index:]
  return training_data, testing_data


def data_contruction(all_day_k, short_lookback_period, mid_lookback_period, long_lookback_period, Label_Bull, Label_Bear, short_x, short_y, mid_x, mid_y, long_x, long_y):

  for today_i in range(0, len(all_day_k), 1):
    #get day-k in the past 100-day period and the forward 1-day
    day_k_past = all_day_k[:today_i+1]
    day_k_forward = all_day_k[today_i+1:]
    if len(day_k_past)<long_lookback_period or len(day_k_forward)<1:
      continue
    day_k_past_win_long = day_k_past[-long_lookback_period:]
    day_k_past_win_mid = day_k_past[-mid_lookback_period:]
    day_k_past_win_short = day_k_past[-short_lookback_period:]
    day_k_forward_win = day_k_forward[:30]

    day_k_past_win_long = standardize(day_k_past_win_long)
    day_k_past_win_mid = standardize(day_k_past_win_mid)
    day_k_past_win_short = standardize(day_k_past_win_short)

    # day_k_past_win_long = day_k_past_win_long
    # day_k_past_win_mid = day_k_past_win_mid
    # day_k_past_win_short = day_k_past_win_short

    #linear regression with day_k_forward_win
    #get the slope
    # slope = (day_k_forward_win.iloc[-1]["Close"] - day_k_forward_win.iloc[0]["Close"])/len(day_k_forward_win)
    try:
      coef = np.polyfit(range(len(day_k_forward_win.iloc[:]["Close"])),day_k_forward_win.iloc[:]["Close"],1)
      slope = coef[0]

      #find label
      #today_price =day_k_past_win_long.iloc[-1]["Close"]
      #tomorrow_price = day_k_forward_win.iloc[0]["Close"]
      # label = Label_Bull if slope >= 0 else Label_Bear

      label = slope

      # store
      long_x.append(day_k_past_win_long)
      mid_x.append(day_k_past_win_mid)
      short_x.append(day_k_past_win_short)
      long_y.append(label)
      mid_y.append(label)
      short_y.append(label)
    except:
      print(today_i)
      pass

  return short_x, short_y, mid_x, mid_y, long_x, long_y


#loop all the symbols in symbols and download data and construct data
for symbol in symbols:
  all_day_k = download_data(symbol, history)
  all_day_k_filtered=remove_outliers(all_day_k)
  # all_day_k_filtered = all_day_k
  training_data, testing_data = spliting_data(all_day_k_filtered, 0.8)
  train_short_x, train_short_y, train_mid_x, train_mid_y, train_long_x, train_long_y = data_contruction(training_data, short_lookback_period, mid_lookback_period, long_lookback_period, Label_Bull, Label_Bear, train_short_x, train_short_y, train_mid_x, train_mid_y, train_long_x, train_long_y)
  test_short_x, test_short_y, test_mid_x, test_mid_y, test_long_x, test_long_y = data_contruction(testing_data, short_lookback_period, mid_lookback_period, long_lookback_period, Label_Bull, Label_Bear, test_short_x, test_short_y, test_mid_x, test_mid_y, test_long_x, test_long_y)
  print(symbol)
  print("Training dataset Length")
  print(len(train_short_x), len(train_short_y), len(train_mid_x), len(train_mid_y), len(train_long_x), len(train_long_y))
  print("Testing dataset Length")
  print(len(test_short_x), len(test_short_y), len(test_mid_x), len(test_mid_y), len(test_long_x), len(test_long_y))

train_short_x, train_short_y = np.array(train_short_x), np.array(train_short_y)
train_mid_x, train_mid_y = np.array(train_mid_x), np.array(train_mid_y)
train_long_x, train_long_y = np.array(train_long_x), np.array(train_long_y)

test_short_x, test_short_y = np.array(test_short_x), np.array(test_short_y)
test_mid_x, test_mid_y = np.array(test_mid_x), np.array(test_mid_y)
test_long_x, test_long_y = np.array(test_long_x), np.array(test_long_y)

np.savez("train_data.npz", short_x=train_short_x, short_y=train_short_y, mid_x=train_mid_x, mid_y=train_mid_y, long_x=train_long_x, long_y=train_long_y)
np.savez("test_data.npz", short_x=test_short_x, short_y=test_short_y, mid_x=test_mid_x, mid_y=test_mid_y, long_x=test_long_x, long_y=test_long_y)