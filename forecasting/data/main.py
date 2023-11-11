import requests
import pandas as pd
import yfinance as yf


def download_data(symbol, history):
    """
    Download historical stock data for a given symbol and time period.

    Args:
        symbol (str): The stock symbol for which the data is to be downloaded.
        history (str): The time period for which the data is to be downloaded.

    Returns:
        pandas.DataFrame: Processed historical stock data for the given symbol and time period.
    """
    # Download data
    all_day_k = yf.Ticker(symbol).history(period=history, interval="1d")

    # Remove dividend and stock splits column
    all_day_k = all_day_k.drop(columns=["Dividends", "Stock Splits"])

    # Remove latest row as it may be incomplete
    all_day_k = all_day_k[:-1]

    return all_day_k


def main():
    """
    Reads a file named "ticker.txt" which contains a list of stock symbols.
    Iterates over each symbol, calls the `download_data` function to download the historical stock data for the symbol,
    and saves the data as a CSV file with the symbol as the filename.
    """

    with open("ticker.txt", "r") as file:
        # Read each line of the file into a list
        symbols = file.readlines()

    # Remove any newline characters from each list item
    symbols = [item.strip() for item in symbols]

    for symbol in symbols:
        data: pd.DataFrame
        data = download_data(symbol, "10y")
        data.to_csv(f"{symbol}.csv", index=True)


if __name__ == "__main__":
    main()
