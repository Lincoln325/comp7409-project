import numpy as np
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm

from config import Config


class TimeSeriesDataset(Dataset):
    def __init__(self, data, window):
        X: np.ndarray = data[f"{window}_x"]
        self.x = X.transpose((0,2,1))

        # short = data["short_x"]
        # mid = data["mid_x"]
        # long = data["long_x"]

        # short = self.process(short)
        # mid = self.process(mid)
        # long = self.process(long)
        # self.x = np.concatenate((short, mid, long),axis=1)
        self.y = data[f"{window}_y"]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # x = self.x[idx]
        # x = x/np.linalg.norm(x)
        # x = np.expand_dims(x, 0)
        return self.x[idx], self.y[idx]

    @staticmethod
    def process(data):
        dim = 10
        m = np.empty((data.shape[0], 1, 5, dim))
        for i, x in tqdm(enumerate(data)):
            n = np.empty((5, dim))
            # scaler = StandardScaler()
            # scaler.fit(x)
            # x = scaler.transform(x)
            for j, v in enumerate(x.T):
                cs = CubicSpline(range(len(v)), v)
                v = cs(np.array(list(range(dim))) * len(v) / dim)
                # n[j] = (v - v.mean())/v.std()
                n[j] = v
            n = np.expand_dims(n, 0)
            m[i] = n
        return m

def create_dataloader(window):
    train_data = np.load("train_data.npz")
    validation_data = np.load("test_data.npz")

    # Create the dataset
    train_dataset = TimeSeriesDataset(train_data, window)
    validation_dataset = TimeSeriesDataset(validation_data, window)

    # Create the dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    validation_dataloder = DataLoader(validation_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    return train_dataloader, validation_dataloder

if __name__ == "__main__":
    # print(validation_dataset[0][0])
    pass
