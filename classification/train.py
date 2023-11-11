import numpy as np
import torch
from torch import nn

from model import TimeSeriesBinaryClassificationModel
from datasets import create_dataloader

from config import Config
from utlis import create_result_dir, write_result_to_csv, compute_recall_precsion

model = TimeSeriesBinaryClassificationModel(Config.WINDOW, Config.FEATURES)

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, 100, 0.95)
criterion = nn.BCEWithLogitsLoss()

result_dir = create_result_dir()
columns = [
    "Epoch",
    "Learning Rate",
    "Training Loss",
    "Validation Loss",
    "Training Recall",
    "Training Precision",
    "Training F1 Score",
    "Validation Recall",
    "Validation Precision",
    "Validation F1 Score",
]

results_overall = "results_overall"
results_bull = "results_bull"
results_bear = "results_bear"

for filename in [results_overall, results_bull, results_bear]:
    write_result_to_csv(
        result_dir,
        filename,
        columns,
    )

metrics = [
    "bull_recall",
    "bear_recall",
    "bull_precision",
    "bear_precision",
    "bull_f1_score",
    "bear_f1_score",
]

train_dataloader, validation_dataloder = create_dataloader(Config.WINDOW)

for epoch in range(Config.EPOCHS):  # 10 epochs for example
    running_loss = 0.0
    running_metrics = {k: 0.0 for k in metrics}

    lr = scheduler.get_last_lr()[0]
    print("EPOCH LR: ", lr)
    for i, (inputs, targets) in enumerate(train_dataloader):
        inputs = inputs.float()
        targets = targets.float().unsqueeze(1)
        # print(inputs)

        optimizer.zero_grad()

        outputs = model(inputs)
        # print((outputs > 0.5).float())
        # print(outputs)
        # print(((outputs > 0.5).float() == targets).float().sum() / len(targets))
        # print((outputs > 0.5).float())
        # print(targets)
        # print(targets)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(f"EPOCH: {epoch} ITERATION: {i+1} TRAIN LOSS: {loss.item()}")

        for metric_name, metric_value in zip(
            metrics,
            compute_recall_precsion((outputs > 0.5).float(), targets),
        ):
            running_metrics[metric_name] += metric_value.item()

    scheduler.step()

    (
        training_loss,
        training_bull_recall,
        training_bear_recall,
        training_bull_precision,
        training_bear_precision,
        training_bull_f1_score,
        training_bear_f1_score,
    ) = [
        metric / len(train_dataloader)
        for metric in (running_loss, *list(running_metrics.values()))
    ]

    running_loss = 0.0
    running_metrics = {k: 0.0 for k in metrics}

    for i, (inputs, targets) in enumerate(validation_dataloder):
        with torch.no_grad():
            inputs = inputs.float()
            targets = targets.float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            for metric_name, metric_value in zip(
                metrics,
                compute_recall_precsion((outputs > 0.5).float(), targets),
            ):
                running_metrics[metric_name] += metric_value.item()

    (
        validation_loss,
        validation_bull_recall,
        validation_bear_recall,
        validation_bull_precision,
        validation_bear_precision,
        validation_bull_f1_score,
        validation_bear_f1_score,
    ) = [
        metric / len(validation_dataloder)
        for metric in (running_loss, *list(running_metrics.values()))
    ]

    print(
        f"EPOCH {epoch} AVG TRAIN LOSS: {training_loss} AVG VALIDATION LOSS {validation_loss}"
    )

    write_result_to_csv(
        result_dir,
        results_overall,
        [
            epoch,
            lr,
            training_loss,
            validation_loss,
            (training_bull_recall + training_bear_recall) / 2,
            (training_bull_precision + training_bear_precision) / 2,
            (training_bull_f1_score + training_bear_f1_score) / 2,
            (validation_bull_recall + validation_bear_recall) / 2,
            (validation_bull_precision + validation_bear_precision) / 2,
            (validation_bull_f1_score + validation_bear_f1_score) / 2,
        ],
    )

    write_result_to_csv(
        result_dir,
        results_bull,
        [
            epoch,
            lr,
            training_loss,
            validation_loss,
            training_bull_recall,
            training_bull_precision,
            training_bull_f1_score,
            validation_bull_recall,
            validation_bull_precision,
            validation_bull_f1_score,
        ],
    )

    write_result_to_csv(
        result_dir,
        results_bull,
        [
            epoch,
            lr,
            training_loss,
            validation_loss,
            training_bear_recall,
            training_bear_precision,
            training_bear_f1_score,
            validation_bear_recall,
            validation_bear_precision,
            validation_bear_f1_score,
        ],
    )
