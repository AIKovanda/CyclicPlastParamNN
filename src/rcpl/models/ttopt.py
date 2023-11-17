import numpy as np
import torch
from ttopt import TTOpt


class TTOptModel:
    def __init__(self, kappa_dim, dim, rank, ttopt_params=None, epsp=None):
        self.d = 1 + kappa_dim + 2 * dim  # Number of function dimensions:
        self.rank = rank  # Maximum TT-rank while cross-like iterations
        self.kappa_dim = kappa_dim
        self.dim = dim
        self.epsp = None if epsp is None else np.array(epsp)
        self.ttopt_params = ttopt_params if ttopt_params is not None else {}

    def train(self):
        pass

    def eval(self):
        pass

    def load_state_dict(self, state_dict):
        pass

    def to(self, device):
        return self

    def random_cyclic_plastic_loading_theta(self, theta, epsp):
        return random_cyclic_plastic_loading(
            k0=theta[:1],
            kap=theta[1:1 + self.kappa_dim],
            c=theta[1 + self.kappa_dim:1 + self.kappa_dim + self.dim],
            a=theta[1 + self.kappa_dim + self.dim:],
            epsp=epsp,
        )

    def __call__(self, x: torch.Tensor):
        np.random.seed(42)
        x_npy = x.cpu().numpy()
        predicted = np.zeros((x_npy.shape[0], 1 + self.kappa_dim + 2 * self.dim))

        for item_id in range(x_npy.shape[0]):

            if self.epsp is None:
                epsp = x_npy[item_id, 1, :]
            else:
                epsp = self.epsp

            def to_optimize(thetas):
                res = np.zeros((thetas.shape[0], x_npy.shape[-1]))
                for i, theta in enumerate(thetas):
                    res[i] = self.random_cyclic_plastic_loading_theta(theta, epsp)
                return np.mean((res - x_npy[item_id, 0, :]) ** 2, axis=1)

            tto = TTOpt(
                f=to_optimize,  # Function for minimization. X is [samples, dim]
                d=self.d,  # Number of function dimensions
                **self.ttopt_params,
                x_opt_real=np.ones(self.d),  # Real value of x-minima (x; this is for test)
                with_log=False)
            tto.optimize(self.rank)
            predicted[item_id] = tto.x_opt

        return torch.from_numpy(predicted).float().to(x.device)

