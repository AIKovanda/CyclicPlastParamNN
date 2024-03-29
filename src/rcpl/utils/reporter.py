import pickle

import numpy as np


class Reporter:
    def __init__(self, store_limit=0, **meta):
        self._stats = {**meta, 'stats': {}}
        self.mean_sum = {}
        self.mean_counts = {}

        self.store_limit = store_limit
        self.stored_data = {}
        self.stored_count = {}

    def get_mean(self, key=None, reset=True):
        if key is None:
            res = {key: self.mean_sum[key] / self.mean_counts[key] for key in self.mean_sum.keys()}
            if reset:
                self.mean_sum = {}
                self.mean_counts = {}
            return res
        res = self.mean_sum[key] / self.mean_counts[key]
        if reset:
            self.mean_sum[key] = 0
            self.mean_counts[key] = 0
        return res

    def get_stored(self, key=None, reset=True):
        if key is None:
            if reset:
                res = self.stored_data
                self.stored_data = {}
                self.stored_count = {}
                return res
            else:
                return self.stored_data.copy()

        res = np.mean(self.stored_data[key], axis=0)
        if reset:
            self.stored_data[key] = []
            self.stored_count[key] = 0
        return res

    def report(self):
        return ", ".join([f"{key}: {self.get_mean(key):.4f}" for key in self.mean_sum.keys()])

    def add_scalar(self, key, value, id_):
        assert isinstance(value, (int, float))
        if key not in self._stats['stats']:
            self._stats['stats'][key] = {}
        self._stats['stats'][key][id_] = value

    def add_data(self, key, value, do_print=False, use_mean=True):
        if do_print:
            print(key, value)
        if key not in self.mean_sum:
            self.mean_sum[key] = 0
            self.mean_counts[key] = 0
        self.mean_sum[key] += np.mean(value)
        self.mean_counts[key] += 1

        if self.store_limit > 0:
            if key not in self.stored_data:
                self.stored_data[key] = []
                self.stored_count[key] = 0
            if use_mean:
                self.stored_data[key].append(np.mean(value))
            else:
                self.stored_data[key].append(value)
            self.stored_count[key] += 1
            if self.stored_count[key] > self.store_limit:
                self.stored_data[key].pop(0)
                self.stored_count[key] -= 1

    def save(self, pickle_path):
        with open(pickle_path, 'wb') as f:
            pickle.dump(self._stats, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    reporter = Reporter(batch_size=2, deque_max_len=30)
    for i in range(1, 7):
        reporter.add_data('Loss/val/_all', i)
    arr = np.zeros(10)
    for i in range(10):
        random_arr = np.random.rand(10)
        arr = arr + random_arr
        reporter.add_data('Loss/val/arr', random_arr)

    assert reporter.get_mean('Loss/val/_all') == 3.5

    assert np.allclose(reporter.get_mean('Loss/val/arr'), np.mean(arr) / 10), (reporter.get_mean('Loss/val/arr'), np.mean(arr) / 10)
