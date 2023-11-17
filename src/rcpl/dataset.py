from torch.utils.data import Dataset


class CPLDataset(Dataset):
    def __init__(self, x, y, epsp=None):
        self.x = x
        self.y = y
        self.epsp = epsp

    def __getitem__(self, index):
        if self.epsp is None:
            return self.x[index], self.y[index]
        return self.x[index], self.y[index], self.epsp

    def __len__(self):
        return self.x.shape[0]
