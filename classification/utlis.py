import os

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def write_result_to_csv(result_dir, filename, metrics: list):
    with open(os.path.join(result_dir, f"{filename}.csv"),'a') as f:
        metrics = map(str, metrics) 
        f.write(",".join(metrics))
        f.write('\n')

def create_result_dir():
    i = 1
    while True:
        path = f"./runs/run_{i}"
        if not os.path.isdir(path):
            os.makedirs(path)
            break
        i += 1
    return path

def compute_recall_precsion(preds, targets):
    from torchmetrics.classification import Recall, Precision, F1Score

    recall = Recall(task="multiclass", average="none", num_classes=2)
    precision = Precision(task="multiclass", average="none", num_classes=2)
    f1_score = F1Score(task="multiclass", average="none", num_classes=2)

    bull_recall, bear_recall = recall(preds, targets)
    bull_precision, bear_precision = precision(preds, targets)
    bull_f1_score, bear_f1_score = f1_score(preds, targets)

    return (
        bull_recall,
        bear_recall,
        bull_precision,
        bear_precision,
        bull_f1_score,
        bear_f1_score,
    )

def standardize(data: pd.DataFrame):
#   scaler = StandardScaler()
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)