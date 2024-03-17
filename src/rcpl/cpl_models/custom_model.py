from abc import ABC

import numpy as np
import torch
from numba import jit
from taskchain.parameter import AutoParameterObject

from rcpl.cpl_models import CPLModel, CPLModelFactory


class CustomModel(CPLModel, ABC):  # This is a custom_model for a new cyclic plastic loading model
    model_name = 'CustomModelName'

    def __init__(
            self,
            theta: np.ndarray | torch.Tensor,  # this is the model parameter vector to be optimized
            custom_parameter=1,  # parameters that are specific to the model and are not optimized
            # add more parameters here
    ):
        self.custom_parameter = custom_parameter
        super().__init__(theta=theta)

    @property
    def theta_len(self):
        # Change this to return the length of the parameter vector.
        return 1 + self.custom_parameter  # This is just an example.

    def _get_labels(self, latex=False) -> list[str]:
        # This function returns the labels for the model parameters.
        if latex:
            return [r'$\alpha$'] + [fr'$\beta_{i+1}$' for i in range(self.custom_parameter)]  # This is just an example.
        return ['alpha'] + [f'beta{i+1}' for i in range(self.custom_parameter)]  # This is just an example.

    def _predict_stress(self, signal: np.ndarray) -> np.ndarray:
        assert self.theta[0] > 0, 'Make a check for the parameters here'
        return super_fast_numpy_implementation(
            alpha=self.theta[:1],
            beta=self.theta[1:1 + self.custom_parameter],
            epsp=signal.flatten(),
        )

    def _predict_stress_torch_batch(self, signal: torch.Tensor) -> torch.Tensor:
        assert torch.all(self.theta[:, 0] > 0), 'Make a check for the parameters here'
        return torch_batch_implementation(
            alpha=self.theta[:, :1],
            beta=self.theta[:, 1:1 + self.custom_parameter],
            epsp=signal[:, 0, :],
        )

    def _predict_stress_torch(self, signal: torch.Tensor) -> torch.Tensor:
        assert self.theta[0] > 0, 'Make a check for the parameters here'
        return torch_batch_implementation(  # This makes a batch with a single sample and calls the batch implementation
            alpha=torch.unsqueeze(self.theta[:1], 0),
            beta=torch.unsqueeze(self.theta[1:1 + self.custom_parameter], 0),
            epsp=torch.unsqueeze(signal, 0),
        )[0]


class CustomModelFactory(AutoParameterObject, CPLModelFactory):

    def make_random_theta(self):
        """
        Generates random parameters for the model.
        """
        custom_parameter = self.model_kwargs['custom_parameter']
        theta = np.zeros(custom_parameter + 1)
        theta[0] = np.random.uniform(*self.params_bound['alpha'])
        for i in range(custom_parameter):
            theta[i + 1] = np.random.uniform(*self.params_bound[f'beta{i+1}'])
        return theta

    def make_model(self, theta: np.ndarray | torch.Tensor = None):
        # Modify this but keep the logic.
        model = CustomModel(
            theta=theta if theta is not None else self.make_random_theta(),
            **self.model_kwargs,
        )
        assert self.labels == model.labels(latex=False), f'Labels do not match {self.labels=} {model.labels(latex=False)=}'
        return model


@jit(nopython=True)  # this makes the function much faster by compiling it to machine code
def super_fast_numpy_implementation(alpha: np.ndarray, beta: np.ndarray, epsp: np.ndarray):
    new_epsp = epsp.copy()
    new_epsp[::2] *= alpha
    new_epsp[1::2] *= beta[0]
    return new_epsp  # Replace this with the actual model. Use numpy functions for faster computation.


def torch_batch_implementation(alpha: torch.Tensor, beta: torch.Tensor, epsp: torch.Tensor):
    new_epsp = epsp.clone()
    new_epsp[:, ::2] *= alpha
    new_epsp[:, 1::2] *= beta[:, :1]
    return new_epsp  # Replace this with the actual model.
