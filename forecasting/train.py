import os
import math
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from datasets import preprocess, load_data
from model import V1
from config import Config
from utlis import create_result_dir, write_result_to_csv


def evaluate(scaler, y_train, y_train_pred, y_test, y_test_pred):
    """
    Calculate the root mean squared error (RMSE) scores for the training and testing data.

    Args:
        scaler: The scaler object used to transform the data.
        y_train: The actual values of the training data.
        y_train_pred: The predicted values of the training data.
        y_test: The actual values of the testing data.
        y_test_pred: The predicted values of the testing data.

    Returns:
        trainScore: The RMSE score for the training data.
        testScore: The RMSE score for the testing data.
    """
    y_train, y_train_pred, y_test, y_test_pred = inverse_transform(
        scaler, y_train, y_train_pred, y_test, y_test_pred
    )
    trainScore = math.sqrt(mean_squared_error(y_train[:, 0], y_train_pred[:, 0]))
    testScore = math.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0]))

    return trainScore, testScore


def inverse_transform(scaler, y_train, y_train_pred, y_test, y_test_pred):
    """
    Reverse the scaling transformation applied to the target variables to calculate the RMSE scores for the training and testing data.

    Args:
        scaler: The scaler object used to transform the data.
        y_train: The actual values of the training data (scaled).
        y_train_pred: The predicted values of the training data.
        y_test: The actual values of the testing data (scaled).
        y_test_pred: The predicted values of the testing data.

    Returns:
        trainScore: The RMSE score for the training data.
        testScore: The RMSE score for the testing data.
    """

    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test.detach().numpy())

    return y_train, y_train_pred, y_test, y_test_pred


def plot_result(name: str, save_dir, data, y_test, y_test_pred):
    """
    Plot the real and predicted stock prices for a given stock name.

    Args:
        name (str): The name of the stock.
        save_dir (str): The directory to save the plot.
        data (DataFrame): The raw data containing the stock prices.
        y_test (ndarray): The actual stock prices.
        y_test_pred (ndarray): The predicted stock prices.

    Returns:
        None. The function only plots and saves the stock price plot.
    """
    _, ax = plt.subplots(figsize=(15, 6))
    ax.xaxis_date()

    ax.plot(
        data[len(data) - len(y_test) :].index,
        y_test,
        color="red",
        label=f"Real {name} Stock Price",
    )
    ax.plot(
        data[len(data) - len(y_test) :].index,
        y_test_pred,
        color="blue",
        label=f"Predicted {name} Stock Price",
    )

    plt.title(f"{name} Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel(f"{name} Stock Price")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{name.lower()}_pred.png"))


def main():
    """
    The `main` function is the entry point of the code.
    It performs the main training loop for a machine learning model using stock market data.
    It expects the data files to be present in the "data" directory.
    The function creates a directory to save the results, loads the data from the CSV files, preprocesses it, trains the model, and saves the training and validation loss and RMSE scores to a CSV file.
    """

    result_dir = create_result_dir()
    with open(os.path.join(result_dir, "config.json"), "w") as f:
        config = Config()
        json.dump(
            dict(
                (name, getattr(config, name))
                for name in dir(config)
                if not name.startswith("__")
            ),
            f,
        )

    for file in os.listdir("data"):
        if "csv" in file:
            save_dir = os.path.join(result_dir, file[:-4])
            os.mkdir(save_dir)

            write_result_to_csv(
                save_dir,
                "results",
                [
                    "Epoch",
                    "Training Loss",
                    "Validation Loss",
                    "Training RMSE",
                    "Validation RMSE",
                ],
            )

            data_raw = pd.read_csv(f"./data/{file}")
            data_raw["Date"] = pd.to_datetime(data_raw["Date"])
            data_raw = data_raw.set_index("Date")
            data, scaler = preprocess(data_raw)

            features_index = {"Open": 0, "High": 1, "Low": 2, "Close": 3, "Volume": 4}
            features = [features_index.get(feat) for feat in Config.FEATURES]
            x_train, y_train, x_test, y_test = load_data(
                data[:, features], data[:, 3].reshape(-1, 1), Config.LOOK_BACK
            )

            model = V1(input_dim=len(features))
            loss_fn = torch.nn.MSELoss()
            optimiser = torch.optim.Adam(model.parameters(), lr=Config.LR)

            print(f"Started training for {file[:-4]}")
            for t in range(Config.EPOCHS):
                y_train_pred = model(x_train)

                loss = loss_fn(y_train_pred, y_train)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                y_test_pred = model(x_test)
                loss_val = loss_fn(y_test_pred, y_test)

                if t % 10 == 0 and t != 0:
                    print(
                        "Epoch ",
                        t,
                        "TRAIN MSE: ",
                        loss.item(),
                        "VALID MSE: ",
                        loss_val.item(),
                    )

                trainScore, testScore = evaluate(
                    scaler, y_train, y_train_pred, y_test, y_test_pred
                )

                write_result_to_csv(
                    save_dir,
                    "results",
                    [t, loss.item(), loss_val.item(), trainScore, testScore],
                )

                if t == (Config.EPOCHS - 1):
                    _, _, y_test, y_test_pred = inverse_transform(
                        scaler, y_train, y_train_pred, y_test, y_test_pred
                    )

                    plot_result(
                        file[:-4],
                        save_dir,
                        data_raw,
                        y_test,
                        y_test_pred,
                    )


if __name__ == "__main__":
    main()
