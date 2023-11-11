import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch


def preprocess(data: pd.DataFrame):
    """
    Preprocesses the input data by filling missing values using forward filling, scaling the features using MinMaxScaler, and returning the scaled features and the scaler object.

    Args:
        data (pd.DataFrame): The input data as a pandas DataFrame.

    Returns:
        features (ndarray): The scaled features as a 2D array.
        scaler (MinMaxScaler): The scaler object used to scale the features.
    """
    data = data.ffill()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    features = scaler.fit_transform(data.values.reshape(-1, len(data.columns)))
    scaler.fit(data["Close"].values.reshape(-1, 1))

    return features, scaler


def load_data(data_features, data_label, look_back):
    """
    Preprocesses the data by splitting it into training and testing sets, and then converts the data into torch tensors.
    
    Args:
        data_features (numpy.ndarray): A numpy array containing the features of the data.
        data_label (numpy.ndarray): A numpy array containing the labels of the data.
        look_back (int): An integer representing the number of previous time steps to use as input for predicting the next time step.
    
    Returns:
        list: A list containing the training data, training labels, testing data, and testing labels as torch tensors.
    """
    data = []
    labels = []
    
    for index in range(len(data_features) - look_back):
        data.append(data_features[index : index + look_back])
        labels.append(data_label[index:index + look_back])

    data = np.array(data)
    labels = np.array(labels)
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[:train_set_size, :-1, :]
    y_train = labels[:train_set_size, -1, :]

    x_test = data[train_set_size:, :-1]
    y_test = labels[train_set_size:, -1, :]

    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    return [x_train, y_train, x_test, y_test]