import torch
from torch import nn
from torch.nn import (
    Module,
    init,
    Linear,
    LSTM,
    Conv1d,
    LeakyReLU,
    Dropout,
    AdaptiveAvgPool1d,
    ModuleList,
)


class LSTMEncoder(Module):
    def __init__(self, dims, **kwargs) -> None:
        super().__init__()
        self.lstm1 = LSTM(10, 1024, num_layers=4, batch_first=True)
        self.conv1 = Conv1d(dims, 1, 10, padding="same")
        self.relu1 = LeakyReLU()

        for m in self.modules():
            if isinstance(m, LSTM):
                for name, param in m.named_parameters():
                    if "bias" in name:
                        init.constant_(param, 0.0)
                    elif "weight" in name:
                        init.xavier_normal_(param, 10)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.conv1(x)
        x = self.relu1(x)

        return x


class TimeSeriesBinaryClassificationModel(Module):
    def __init__(self, windows, features):
        super(TimeSeriesBinaryClassificationModel, self).__init__()
        self.features = self._get_dims(
            {"o": 0, "h": 1, "l": 2, "c": 3, "v": 4}, features
        )

        self.windows = self._get_dims({"short": 0, "mid": 1, "long": 2}, windows)

        self.encoders = ModuleList(
            [LSTMEncoder(len(self.features)) for _ in range(len(windows))]
        )

        self.pooling = AdaptiveAvgPool1d(2000)
        self.fc = Linear(2000, 512)
        self.dropout = Dropout(0.2)
        self.head = Linear(512, 1)

        for m in self.modules():
            if isinstance(m, Linear):
                init.xavier_normal_(m.weight, 10)
                init.constant_(m.bias, 0)

    @staticmethod
    def _get_dims(dims_map, keys):
        return [dims_map.get(key) for key in keys]

    def forward(self, x):
        x_latent = []
        for encoder, window in zip(self.encoders, self.windows):
            x_latent.append(encoder(x[:, window, self.features, :]))
        
        x = torch.cat(x_latent, dim=1)
        x = x.reshape(x.size()[0], -1)

        x = self.pooling(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.head(x)

        return x


if __name__ == "__main__":
    model = TimeSeriesBinaryClassificationModel()
    print(model(torch.rand(3, 2, 30)))
