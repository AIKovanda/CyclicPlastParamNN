import numpy as np
import torch
from taskchain.parameter import AutoParameterObject
from ttopt import TTOpt

from rcpl.utils.simplex import to_optimize_one


class TTOptModel(AutoParameterObject):
    def __init__(self, rank, ttopt_params=None, epsp=None):
        self.rank = rank  # Maximum TT-rank while cross-like iterations
        self.epsp = None if epsp is None else np.array(epsp)
        self.ttopt_params = ttopt_params if ttopt_params is not None else {}
        self.model_factory = None

    def train(self):
        pass

    def eval(self):
        pass

    def load_state_dict(self, state_dict):
        pass

    def to(self, device):
        return self

    def __call__(self, x: torch.Tensor):
        np.random.seed(42)
        x_npy = x.cpu().numpy()
        predicted = np.zeros((x_npy.shape[0], self.model_factory.num_params))

        for item_id in range(x_npy.shape[0]):

            if self.epsp is None:
                epsp = x_npy[item_id, 1:2, :]
            else:
                epsp = self.epsp

            def to_optimize(thetas):

                res = np.zeros(thetas.shape[0])
                for i, theta in enumerate(thetas):
                    res[i] = to_optimize_one(
                        unscaled_theta=theta,
                        true_stress=x_npy[item_id, 0:1, :],
                        signal_=epsp,
                        model_factory=self.model_factory,
                    )
                return res

            tto = TTOpt(
                f=to_optimize,  # Function for minimization. X is [samples, dim]
                d=self.model_factory.num_params,  # Number of function dimensions
                a=self.model_factory.lower_bound,  # Left bound of the function domain
                b=self.model_factory.upper_bound,  # Right bound of the function domain
                **self.ttopt_params,
                x_opt_real=np.ones(self.model_factory.num_params),  # Real value of x-minima (x; this is for test)
                with_log=False)
            tto.optimize(self.rank)
            predicted[item_id] = tto.x_opt

        return torch.from_numpy(predicted).float().to(x.device)
