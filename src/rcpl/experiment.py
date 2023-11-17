from pathlib import Path

import numpy as np
import torch
from taskchain.parameter import AutoParameterObject


class Experiment(AutoParameterObject):

    def __init__(self, epsp: list | np.ndarray | torch.Tensor | Path, stress: list | np.ndarray | torch.Tensor | Path = None,
                 representation: tuple = None):
        self.epsp = epsp
        self.stress = stress
        self.representation = tuple(representation) if representation is not None else ('raw',)

        assert epsp is not None
        epsp = self._load_to_numpy(epsp)
        assert len(epsp.shape) == 1
        stress = self._load_to_numpy(stress)
        if stress is not None:
            assert len(stress.shape) == 1

        self.epsp_representation = {self.representation: epsp}
        self.stress_representation = {self.representation: stress}

    def _load_to_numpy(self, data):
        if isinstance(data, list):
            return np.array(data)
        elif isinstance(data, Path):
            if data.suffix == '.npy':
                return np.load(data)
            else:
                raise NotImplementedError
        return data

    def get_epsp_representation(self, definition: tuple):
        if definition[0] == 'raw':
            return self.epsp_representation[('raw',)]
        elif definition[0] == 'geom':
            return self.geometrically_interpolated_epsp(*definition[1:])
        elif definition[0] == 'lin':
            return self.linearly_interpolated_epsp(*definition[1:])
        else:
            raise NotImplementedError

    def get_stress_representation(self, definition: tuple):
        if definition[0] == 'raw':
            return self.stress_representation[('raw',)]
        elif definition[0] == 'geom':
            return self.geometrically_interpolated_stress(*definition[1:])
        elif definition[0] == 'lin':
            return self.linearly_interpolated_stress(*definition[1:])
        else:
            raise NotImplementedError

    def get_stress_epsp_representation(self, definition: tuple):
        if definition[0] == 'raw':
            return np.stack([
                self.epsp_representation[('raw',)],
                self.stress_representation[('raw',)],
            ])
        elif definition[0] == 'geom':
            return np.stack([
                self.geometrically_interpolated_stress(*definition[1:]),
                self.geometrically_interpolated_epsp(*definition[1:]),
            ])
        elif definition[0] == 'lin':
            return np.stack([
                self.linearly_interpolated_stress(*definition[1:]),
                self.linearly_interpolated_epsp(*definition[1:]),
            ])
        else:
            raise NotImplementedError

    def geometrically_interpolated_epsp(self, points_per_segment: int, first_last_ratio: float) -> np.ndarray:
        if len(self.epsp_representation) > 0:
            return self.epsp_representation[('geom', points_per_segment, first_last_ratio)]
        raise NotImplementedError

    def geometrically_interpolated_stress(self, points_per_segment: int, first_last_ratio: float) -> np.ndarray:
        if len(self.epsp_representation) > 0:
            return self.stress_representation[('geom', points_per_segment, first_last_ratio)]
        raise NotImplementedError

    def linearly_interpolated_epsp(self, points_per_segment: int) -> np.ndarray:
        if points_per_segment in self.epsp_representation:
            return self.epsp_representation[('lin', points_per_segment)]
        raise NotImplementedError

    def linearly_interpolated_stress(self, points_per_segment: int) -> np.ndarray:
        if points_per_segment in self.stress_representation:
            return self.stress_representation[('lin', points_per_segment)]
        raise NotImplementedError


class RandomExperimentGenerator(AutoParameterObject):

    def __init__(self, epsp_uniform_params: tuple[int, int, int] = None):
        self.epsp_uniform_params = epsp_uniform_params

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

        epsp_r = np.append(0, np.random.uniform(*self.epsp_uniform_params))
        epsp_r[::2] *= -1

        r = first_last_ratio ** (1 / (points_per_segment - 2))  # Ratio between steps.
        geospace = np.array([r ** i for i in range(1, points_per_segment)])
        geospace /= np.sum(geospace)
        geospace = np.cumsum(geospace)
        epsp = [0]
        for segment_id in range(self.epsp_uniform_params[-1]):
            epsp.extend(
                [epsp_r[segment_id + 1] * geo_val + (1 - geo_val) * epsp_r[segment_id] for geo_val in geospace])
        epsp = np.array(epsp)

        return Experiment(epsp, representation=('geom', points_per_segment, first_last_ratio))

    def generate_by_linearly_interpolated_epsp(self, points_per_segment: int) -> Experiment:

        epsp_r = np.append(0, np.random.uniform(*self.epsp_uniform_params))
        epsp_r[::2] *= -1

        epsp = [np.array([0])]
        for segment_id in range(self.epsp_uniform_params[-1]):
            epsp.append(np.linspace(epsp_r[segment_id], epsp_r[segment_id + 1], points_per_segment + 1)[1:])
        epsp = np.concatenate(epsp)

        return Experiment(epsp, representation=('lin', points_per_segment))
