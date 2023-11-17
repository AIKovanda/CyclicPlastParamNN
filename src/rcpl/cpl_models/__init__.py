import abc

import numpy as np
import torch


class CPLModel:

    def __init__(self, theta: np.ndarray | torch.Tensor = None):
        assert theta is not None
        assert theta.ndim == 1
        self.theta = theta
        assert len(theta) == self.theta_len
        self._labels = {}

    def labels(self, latex=False) -> list[str]:  # @cache wasted a lot of memory - this is a workaround
        if latex and 'latex' in self._labels:
            return self._labels['latex']
        elif not latex and 'plain' in self._labels:
            return self._labels['plain']
        labels = self._get_labels(latex=latex)
        self._labels['latex' if latex else 'plain'] = labels
        return labels

    @abc.abstractmethod
    def _get_labels(self, latex=False) -> list[str]:
        pass

    @property
    @abc.abstractmethod
    def theta_len(self) -> int:
        pass

    def predict_stress(self, epsp: np.ndarray) -> np.ndarray:
        assert isinstance(epsp, np.ndarray)
        assert isinstance(self.theta, np.ndarray)
        return self._predict_stress(epsp=epsp)

    @abc.abstractmethod
    def _predict_stress(self, epsp: np.ndarray) -> np.ndarray:
        pass

    def predict_stress_torch(self, epsp: torch.Tensor) -> torch.Tensor:
        assert isinstance(epsp, torch.Tensor)
        assert isinstance(self.theta, torch.Tensor)
        return self._predict_stress_torch(epsp=epsp)

    @abc.abstractmethod
    def _predict_stress_torch(self, epsp: torch.Tensor) -> torch.Tensor:
        pass

    def predict_stress_torch_batch(self, epsp: torch.Tensor) -> torch.Tensor:
        assert isinstance(epsp, torch.Tensor)
        assert isinstance(self.theta, torch.Tensor)
        return self._predict_stress_torch_batch(epsp=epsp)

    @abc.abstractmethod
    def _predict_stress_torch_batch(self, epsp: torch.Tensor) -> torch.Tensor:
        pass


class CPLModelFactory:
    def __init__(self, params_bound: dict, model_kwargs: dict = None):
        self.params_bound = params_bound
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}

    @property
    def lower_bound(self) -> np.ndarray:
        return np.array([v[0] for v in self.params_bound.values()])

    @property
    def upper_bound(self) -> np.ndarray:
        return np.array([v[1] for v in self.params_bound.values()])

    @property
    def labels(self) -> list[str]:
        return list(self.params_bound.keys())

    @abc.abstractmethod
    def make_random_theta(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def make_model(self, theta: np.ndarray | torch.Tensor = None) -> CPLModel:
        pass
