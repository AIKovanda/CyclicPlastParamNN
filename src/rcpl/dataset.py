from torch.utils.data import Dataset


class CPLDataset(Dataset):
    def __init__(self, x, y, signal=None):
        self.x = x
        self.y = y
        self.signal = signal

    def __getitem__(self, index):
        if self.signal is None:
            return self.x[index], self.y[index]
        return self.x[index], self.y[index], self.signal

    def __len__(self):
        return self.x.shape[0]
