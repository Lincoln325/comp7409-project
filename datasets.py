import numpy as np
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from config import Config


class TimeSeriesDataset(Dataset):
    def __init__(self, data, split="train"):
        if split == "train":
            short = data["short_x"][30 - 7:2000+30-7]
            mid = data["mid_x"][30 - 15:2000+30-15]
            long = data["long_x"][:2000]
            y = data["long_y"][:2000]
        else:
            short = data["short_x"][30 - 7+2000:]
            mid = data["mid_x"][30 - 15+2000:]
            long = data["long_x"][2000:]
            y = data["long_y"][2000:]

        short = self.process(short)
        mid = self.process(mid)
        long = self.process(long)
        self.x = np.concatenate((short, mid, long),axis=1)
        self.y = y



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
            for j, v in enumerate(x.T):
                cs = CubicSpline(range(len(v)), v)
                n[j] = cs(np.array(list(range(dim))) * len(v) / dim)
            n = np.expand_dims(n, 0)
            m[i] = n
        return m


data = np.load("datasets.npz")

# Create the dataset
train_dataset = TimeSeriesDataset(data, "train")
validation_dataset = TimeSeriesDataset(data, "valid")

# Create the dataloader
train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
validation_dataloder = DataLoader(validation_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

if __name__ == "__main__":
    print(validation_dataset[0][0].shape)
