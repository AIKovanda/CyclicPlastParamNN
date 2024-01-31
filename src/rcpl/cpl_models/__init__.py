import abc

import numpy as np
import torch


class CPLModel:

    def __init__(self, theta: np.ndarray | torch.Tensor = None):
        assert theta is not None
        assert theta.ndim <= 2  # can be batched
        self.theta = theta
        assert theta.shape[-1] == self.theta_len, f'Expected theta of shape (..., {self.theta_len}), got {theta.shape}'
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

    def predict_stress(self, signal: np.ndarray) -> np.ndarray:
        assert isinstance(signal, np.ndarray)
        assert isinstance(self.theta, np.ndarray)
        assert signal.ndim == 2, f'Expected signal of shape (C, L), got {signal.shape}'
        return self._predict_stress(signal=signal)

    @abc.abstractmethod
    def _predict_stress(self, signal: np.ndarray) -> np.ndarray:
        pass

    def predict_stress_torch(self, signal: torch.Tensor) -> torch.Tensor:
        assert isinstance(signal, torch.Tensor)
        assert isinstance(self.theta, torch.Tensor)
        assert signal.ndim == 2
        return self._predict_stress_torch(signal=signal)

    @abc.abstractmethod
    def _predict_stress_torch(self, signal: torch.Tensor) -> torch.Tensor:
        pass

    def predict_stress_torch_batch(self, signal: torch.Tensor) -> torch.Tensor:
        assert isinstance(signal, torch.Tensor)
        assert isinstance(self.theta, torch.Tensor)
        assert signal.ndim == 3
        return self._predict_stress_torch_batch(signal=signal)

    @abc.abstractmethod
    def _predict_stress_torch_batch(self, signal: torch.Tensor) -> torch.Tensor:
        pass


class CPLModelFactory:
    def __init__(self, params_bound: dict, model_kwargs: dict = None, apriori_distribution_params: dict = None):
        self.params_bound = params_bound
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.apriori_distribution_params = apriori_distribution_params if apriori_distribution_params is not None else {}

    @property
    def lower_bound(self) -> np.ndarray:
        return np.array([v[0] for v in self.params_bound.values()])

    @property
    def simplex_lower_bound(self) -> np.ndarray:
        return np.array([1e-5 for v in self.params_bound.values()])

    @property
    def upper_bound(self) -> np.ndarray:
        return np.array([v[1] for v in self.params_bound.values()])

    @property
    def simplex_upper_bound(self) -> np.ndarray:
        return np.array([np.inf for v in self.params_bound.values()])

    @property
    def labels(self) -> list[str]:
        return list(self.params_bound.keys())

    @abc.abstractmethod
    def make_random_theta(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def make_model(self, theta: np.ndarray | torch.Tensor = None) -> CPLModel:
        pass
