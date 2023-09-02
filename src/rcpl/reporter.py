import pickle
from collections import deque

import numpy as np


class Reporter:
    def __init__(self, batch_size=None, deque_max_len=100, **meta):
        self._stats = {**meta, 'stats': {}}
        self.deque_stats = {}
        self.batch_size = batch_size
        self.deque_max_len = deque_max_len

    def get_mean_deque(self, key=None):
        if key is None:
            return {key: np.mean(value) for key, value in self.deque_stats.items()}
        return np.mean(self.deque_stats[key])

    def report(self):
        return ", ".join([f"{key}: {np.mean(value):.4f}" for key, value in self.deque_stats.items()])

    def add_scalar(self, key, value, id_):
        if key not in self._stats['stats']:
            self._stats['stats'][key] = {}
        self._stats['stats'][key][id_] = value

    def add_deque(self, key, value, is_batched=True, use_max_len=True):
        if key not in self.deque_stats:
            if use_max_len:
                self.deque_stats[key] = deque(maxlen=self.deque_max_len if is_batched else self.deque_max_len * self.batch_size)
            else:
                self.deque_stats[key] = deque()
        self.deque_stats[key].append(value)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self._stats, f, pickle.HIGHEST_PROTOCOL)
