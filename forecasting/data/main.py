import requests
import pandas as pd
import yfinance as yf

def download_data(symbol, history):
  #download data
  all_day_k = yf.Ticker(symbol).history(period=history, interval="1d")

    # Remove dividend and stock splits column
  all_day_k = all_day_k.drop(columns=["Dividends","Stock Splits"])

    # Remove latest row as it may be imcompleted
  all_day_k = all_day_k[:-1]

  return all_day_k

if __name__ == "__main__":
  with open('ticker.txt', 'r') as file:
      # Read each line of the file into a list
      symbols = file.readlines()

  # Remove any newline characters from each list item
  symbols = [item.strip() for item in symbols]

  for symbol in symbols:
    data: pd.DataFrame
    data = download_data(symbol, "10y")
    data.to_csv(f"{symbol}.csv", index=False)
