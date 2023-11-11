import os
import math

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import mean_squared_error

from datasets import preprocess, load_data
from model import V1
from config import Config
from utlis import create_result_dir, write_result_to_csv

def evaluate(scaler, y_train, y_train_pred, y_test, y_test_pred):
    # invert predictions
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test.detach().numpy())

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
    # print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
    # print('Test Score: %.2f RMSE' % (testScore))

    return trainScore, testScore

if __name__ == "__main__":
    result_dir = create_result_dir()

    for file in os.listdir("data"):
        if "csv" in file:
            save_dir = os.path.join(result_dir, file[:-4])
            os.mkdir(save_dir)
            write_result_to_csv(save_dir, "results", ["Epoch", "Training Loss", "Validation Loss", "Training RMSE", "Validation RMSE"])

            data = pd.read_csv(f"./data/{file}")
            data, scaler = preprocess(data)

            features_index = {
                "Open": 0,
                "High": 1,
                "Low": 2,
                "Close": 3,
                "Volume": 4
            }
            features = [features_index.get(feat) for feat in Config.FEATURES]
            x_train, y_train, x_test, y_test = load_data(data[:, features], data[:, 3].reshape(-1,1), Config.LOOK_BACK)

            model = V1(input_dim=len(features))
            loss_fn = torch.nn.MSELoss()
            optimiser = torch.optim.Adam(model.parameters(), lr=Config.LR)

            hist = np.zeros(Config.EPOCHS)

            # Number of steps to unroll
            seq_dim =Config.LOOK_BACK-1

            for t in range(Config.EPOCHS):
                # Initialise hidden state
                # Don't do this if you want your LSTM to be stateful
                #model.hidden = model.init_hidden()
                
                # Forward pass
                y_train_pred = model(x_train)

                loss = loss_fn(y_train_pred, y_train)
                hist[t] = loss.item()

                # Zero out gradient, else they will accumulate between epochs
                optimiser.zero_grad()

                # Backward pass
                loss.backward()

                # Update parameters
                optimiser.step()

                y_test_pred = model(x_test)
                loss_val = loss_fn(y_test_pred, y_test)

                if t % 10 == 0 and t !=0:
                    print("Epoch ", t, "MSE: ", loss.item())

                trainScore, testScore = evaluate(scaler, y_train, y_train_pred, y_test, y_test_pred)

                write_result_to_csv(save_dir, "results", [t, loss.item(), loss_val.item(), trainScore, testScore])