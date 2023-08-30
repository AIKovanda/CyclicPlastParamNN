import pickle
from collections import deque

import numpy as np


class Reporter:
    def __init__(self, deque_x_max_len=100, deque_y_max_len=100, **meta):
        self._stats = {**meta, 'stats': {}}
        self.deque_stats = {}
        self.deque_x_max_len = deque_x_max_len
        self.deque_y_max_len = deque_y_max_len

    def report(self):
        return ", ".join([f"{key}: {np.median(value):.4f}" for key, value in self.deque_stats.items()])

    def add_scalar(self, key, value, id_):
        if key not in self._stats['stats']:
            self._stats['stats'][key] = {}
        self._stats['stats'][key][id_] = value

    def add_deque(self, key, value, type_):
        if key not in self.deque_stats:
            self.deque_stats[key] = deque(maxlen=self.deque_x_max_len if type_ == 'x' else self.deque_y_max_len)
        self.deque_stats[key].append(value)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self._stats, f, pickle.HIGHEST_PROTOCOL)
