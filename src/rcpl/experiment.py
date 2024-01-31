import abc
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta
from taskchain.parameter import AutoParameterObject


class Experiment(AutoParameterObject):

    def __init__(self, signal: list | np.ndarray | torch.Tensor | Path = None, channel_labels: list = None,
                 representation: tuple = None, json_path: Path = None, crop_signal=None, meta: dict = None):
        self.signal = signal
        self.channel_labels = channel_labels
        self.representation = representation
        self.json_path = json_path
        self.representation = tuple(representation) if representation is not None else ('raw',)
        self.crop_signal = crop_signal
        self.meta = meta if meta is not None else {}

        if self.json_path is not None:
            self.needs_loading = True
            self.signal_channel_labels = None
            self._signal_representation = None
        else:
            self.needs_loading = False
            if not isinstance(signal, torch.Tensor):
                signal = np.array(signal)
            if signal.ndim == 1:
                signal = signal[np.newaxis, :]
            assert signal.ndim == 2
            self.signal_channel_labels = channel_labels
            assert len(self.signal_channel_labels) == signal.shape[0]
            self._signal_representation = {self.representation: signal}

    @property
    def signal_representation(self):
        if self.needs_loading:
            self._load_json()
        return self._signal_representation

    def _load_json(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        self._signal_representation = {eval(i): np.array(j) for i, j in data['signal_representation'].items()}
        self.signal_channel_labels = data['channel_labels']
        self.meta = data.get('meta', {})
        self.needs_loading = False

    def get_signal_representation(self, definition: tuple, channels: list = None):
        if self.needs_loading:
            self._load_json()
        if channels is not None:
            assert all([channel in self.signal_channel_labels for channel in channels])

        if definition[0] == 'raw':
            sig = self.signal_representation[('raw',)]
        elif definition[0] == 'geom':
            sig = self.geometrically_interpolated_signal(*definition[1:])
        elif definition[0] == 'lin':
            sig = self.linearly_interpolated_signal(*definition[1:])
        else:
            raise NotImplementedError

        if self.crop_signal is not None:
            sig = sig[:, self.crop_signal[0]:self.crop_signal[1]]
        if channels is None:
            return sig
        channels_id = [self.signal_channel_labels.index(channel) for channel in channels]
        can_be_slicer = len(channels_id) == 1 or all([channels_id[i+1]-channels_id[i] == 1 for i in range(len(channels_id)-1)])
        if can_be_slicer:
            return sig[channels_id[0]:channels_id[-1]+1]
        else:
            return sig[channels_id]

    def geometrically_interpolated_signal(self, points_per_segment: int, first_last_ratio: float) -> np.ndarray:
        if len(self.signal_representation) > 0:
            return self.signal_representation[('geom', points_per_segment, first_last_ratio)]
        raise NotImplementedError

    def linearly_interpolated_signal(self, points_per_segment: int) -> np.ndarray:
        if points_per_segment in self.signal_representation:
            return self.signal_representation[('lin', points_per_segment)]
        raise NotImplementedError


class RandomExperimentGenerator:

    @abc.abstractmethod
    def generate_representation(self, definition: tuple):
        pass


class EpsPRandomExperimentGenerator(AutoParameterObject, RandomExperimentGenerator):

    def __init__(self, epsp_dist_config: dict = None, dist_type='uniform', experiment_kwargs: dict = None):
        self.epsp_dist_config = epsp_dist_config if epsp_dist_config is not None else {}
        self.dist_type = dist_type
        self.experiment_kwargs = experiment_kwargs if experiment_kwargs is not None else {}

    def generate_representation(self, definition: tuple):
        if definition[0] == 'raw':
            raise NotImplementedError
        elif definition[0] == 'geom':
            return self.generate_by_geometrically_interpolated_epsp(*definition[1:])
        elif definition[0] == 'lin':
            return self.generate_by_linearly_interpolated_epsp(*definition[1:])
        else:
            raise NotImplementedError

    def generate_by_geometrically_interpolated_epsp(self, points_per_segment: int, first_last_ratio: float) -> Experiment:
        num_reversals = self.epsp_dist_config['num_reversals']

        if self.dist_type == 'uniform':
            epsp_r = np.append(0, np.random.uniform(*(self.epsp_dist_config['uniform'] + [num_reversals]), size=num_reversals))
            epsp_r[::2] *= -1
        elif self.dist_type == 'beta':
            epsp_r = [0]
            for i, j in enumerate(beta.rvs(self.epsp_dist_config['alpha'], self.epsp_dist_config['beta'],
                                           size=num_reversals)):
                if i > 2 and np.random.rand() < self.epsp_dist_config['return_same_chance']:
                    epsp_r.append(epsp_r[-2])
                else:
                    a = epsp_r[-1] + (-1) ** i * self.epsp_dist_config['min_step']
                    b = (-1) ** i * self.epsp_dist_config['bound']
                    epsp_r.append(a + (b - a) * j)
        else:
            raise NotImplementedError

        r = first_last_ratio ** (1 / (points_per_segment - 2))  # Ratio between steps.
        geospace = np.array([r ** i for i in range(1, points_per_segment)])
        geospace /= np.sum(geospace)
        geospace = np.cumsum(geospace)
        epsp = [0]
        for segment_id in range(num_reversals):
            epsp.extend(
                [epsp_r[segment_id + 1] * geo_val + (1 - geo_val) * epsp_r[segment_id] for geo_val in geospace])
        epsp = np.array(epsp)

        return Experiment(epsp, channel_labels=['epsp'], representation=('geom', points_per_segment, first_last_ratio), **self.experiment_kwargs)

    def generate_by_linearly_interpolated_epsp(self, points_per_segment: int) -> Experiment:

        raise NotImplementedError
        epsp_r = np.append(0, np.random.uniform(*self.epsp_dist_config))
        epsp_r[::2] *= -1

        epsp = [np.array([0])]
        for segment_id in range(self.epsp_dist_config[-1]):
            epsp.append(np.linspace(epsp_r[segment_id], epsp_r[segment_id + 1], points_per_segment + 1)[1:])
        epsp = np.concatenate(epsp)

        return Experiment(epsp, channel_labels=['epsp'], representation=('lin', points_per_segment), **self.experiment_kwargs)
