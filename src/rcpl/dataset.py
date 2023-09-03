import numpy as np
import torch
from torch.utils.data import Dataset

from rcpl.rcpl import Experiment


class RCLPDataset(Dataset):
    def __init__(self, exp: Experiment, purpose='train', dataset_dir=None):
        self.x = torch.from_numpy(np.load(dataset_dir / f'{purpose}_x.npy'))
        self.y = torch.from_numpy(np.load(dataset_dir / f'{purpose}_y.npy')).float()
        self.epsp = exp.epsp

    def __getitem__(self, index):
        return torch.unsqueeze(self.x[index], 0), self.y[index], self.epsp

    def __len__(self):
        return self.x.shape[0]
