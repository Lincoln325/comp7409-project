import numpy as np
import pandas as pd
import yfinance as yf

symbol = "0700.HK"
history = "10y"

all_day_k = yf.Ticker(symbol).history(period=history, interval="1d")

# Remove dividend and stock splits column
all_day_k = all_day_k.drop(columns=["Dividends","Stock Splits"])

# Remove latest row as it may be imcompleted
all_day_k = all_day_k[:-1]

short_lookback_period = 7
mid_lookback_period = 15
long_lookback_period = 30
classes = ["Bull","Bear"]
Label_Bull = classes.index("Bull")
Label_Bear = classes.index("Bear")

short_x, short_y = [],[]
mid_x,mid_y = [], []
long_x,long_y = [], []


for today_i in range(len(all_day_k)):
  #get day-k in the past 100-day period and the forward 1-day
  day_k_past = all_day_k[:today_i+1]
  day_k_forward = all_day_k[today_i+1:]
  if len(day_k_past)<short_lookback_period or len(day_k_forward)<1:
    continue
  day_k_past_win = day_k_past[-short_lookback_period:]
  day_k_forward_win = day_k_forward[:1]

  #find label
  today_price =day_k_past_win.iloc[-1]["Close"]
  tomorrow_price = day_k_forward_win.iloc[0]["Close"]
  label = Label_Bull if tomorrow_price > today_price else Label_Bear

  # store
  short_x.append(day_k_past_win)
  short_y.append(label)

for today_i in range(len(all_day_k)):
  #get day-k in the past 100-day period and the forward 1-day
  day_k_past = all_day_k[:today_i+1]
  day_k_forward = all_day_k[today_i+1:]
  if len(day_k_past)<mid_lookback_period or len(day_k_forward)<1:
    continue
  day_k_past_win = day_k_past[-mid_lookback_period:]
  day_k_forward_win = day_k_forward[:1]

  #find label
  today_price =day_k_past_win.iloc[-1]["Close"]
  tomorrow_price = day_k_forward_win.iloc[0]["Close"]
  label = Label_Bull if tomorrow_price > today_price else Label_Bear

  # store
  mid_x.append(day_k_past_win)
  mid_y.append(label)

for today_i in range(len(all_day_k)):
  #get day-k in the past 100-day period and the forward 1-day
  day_k_past = all_day_k[:today_i+1]
  day_k_forward = all_day_k[today_i+1:]
  if len(day_k_past)<long_lookback_period or len(day_k_forward)<1:
    continue
  day_k_past_win = day_k_past[-long_lookback_period:]
  day_k_forward_win = day_k_forward[:1]

  #find label
  today_price =day_k_past_win.iloc[-1]["Close"]
  tomorrow_price = day_k_forward_win.iloc[0]["Close"]
  label = Label_Bull if tomorrow_price > today_price else Label_Bear

  # store
  long_x.append(day_k_past_win)
  long_y.append(label)

short_x, short_y = np.array(short_x), np.array(short_y)
mid_x, mid_y = np.array(mid_x), np.array(mid_y)
long_x, long_y = np.array(long_x), np.array(long_y)

np.savez("datasets.npz", short_x=short_x, short_y=short_y, mid_x=mid_x, mid_y=mid_y, long_x=long_x, long_y=long_y)