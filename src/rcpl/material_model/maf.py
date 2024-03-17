from abc import ABC

import numpy as np
import torch
from numba import jit
from taskchain.parameter import AutoParameterObject

from rcpl.material_model import CPLModel, CPLModelFactory

SQR32 = np.sqrt(1.5)


class MAFModel(CPLModel, ABC):
    model_name = None

    def __init__(self, theta: np.ndarray | torch.Tensor, dim=4, kappa_dim=2, uses_log_c=False):
        assert self.model_name in ['MAF', 'MAFTr']
        self.dim = dim
        self.kappa_dim = kappa_dim
        self.uses_log_c = uses_log_c
        super().__init__(theta=theta)

    @property
    def theta_len(self):
        return 1 + self.kappa_dim + self.dim * 2 + (1 if self.model_name == 'MAFTr' else 0)

    def _get_labels(self, latex=False):
        label_key = 'latex_label' if latex else 'label'
        uses_params = {
            'k0': {'dim': 1, 'label': 'k0', 'latex_label': 'k_0'},
            'kap': {'dim': self.kappa_dim, 'label': 'κ', 'latex_label': '\\kappa_'},
            'c': {'dim': self.dim, 'label': 'log_c' if self.uses_log_c else 'c', 'latex_label': r'\log c_' if self.uses_log_c else 'c_'},
            'a': {'dim': self.dim, 'label': 'a', 'latex_label': 'a_'},
        }
        if self.model_name == 'MAFTr':
            uses_params['aL'] = {'dim': 1, 'label': 'aL', 'latex_label': '\\overline{a}'}

        labels = []
        for param_info in uses_params.values():
            if param_info['dim'] == 1:
                labels.append(param_info[label_key])
            else:
                for i in range(param_info['dim']):
                    labels.append(f'{param_info[label_key]}{i + 1}')
        return labels


class MAF(MAFModel):
    model_name = 'MAF'

    def _predict_stress(self, signal: np.ndarray) -> np.ndarray:
        assert len(set(signal.shape) - {1}) == 1
        c = self.theta[1 + self.kappa_dim:1 + self.kappa_dim + self.dim]
        if self.uses_log_c:
            c = np.exp(c)
        return random_cyclic_plastic_loading(
            k0=self.theta[:1],
            kap=self.theta[1:1 + self.kappa_dim],
            c=c,
            a=self.theta[1 + self.kappa_dim + self.dim:],
            epsp=signal.flatten(),
        )

    def _predict_stress_torch(self, signal: torch.Tensor) -> torch.Tensor:
        assert len(set(signal.shape) - {1}) == 1
        c = torch.unsqueeze(self.theta[1 + self.kappa_dim:1 + self.kappa_dim + self.dim], 0)
        if self.uses_log_c:
            c = torch.exp(c)
        return random_cyclic_plastic_loading_torch_batch(
            epsp=torch.unsqueeze(signal, 0),
            k0=torch.unsqueeze(self.theta[:1], 0),
            kap=torch.unsqueeze(self.theta[1:1 + self.kappa_dim], 0),
            c=c,
            a=torch.unsqueeze(self.theta[1 + self.kappa_dim + self.dim:], 0),
        )[0]

    def _predict_stress_torch_batch(self, signal: torch.Tensor) -> torch.Tensor:
        assert signal.ndim == 3
        assert signal.shape[1] == 1
        c = self.theta[:, 1 + self.kappa_dim:1 + self.kappa_dim + self.dim]
        if self.uses_log_c:
            c = torch.exp(c)
        return random_cyclic_plastic_loading_torch_batch(
            epsp=signal[:, 0, :],
            k0=self.theta[:, :1],
            kap=self.theta[:, 1:1 + self.kappa_dim],
            c=c,
            a=self.theta[:, 1 + self.kappa_dim + self.dim:],
        )


class MAFCPLModelFactory(AutoParameterObject, CPLModelFactory):
    sort_first = 4
    model_class = MAF

    def make_random_theta(self):
        """
        Generates random parameters for the MAF model.
        """
        dim = self.model_kwargs['dim']
        kappa_dim = self.model_kwargs['kappa_dim']
        uses_log_c = self.model_kwargs.get('uses_log_c', False)
        c_prefix = 'log_' if uses_log_c else ''
        assert kappa_dim % 2 == 0

        a_sum = np.random.uniform(*self.apriori_distribution_params['a_sum_bound'])
        a_lower_bound = np.array([self.params_bound[f'a{i+1}'][0] for i in range(dim)])
        c_lower_bound = np.array([self.params_bound[f'{c_prefix}c{i+1}'][0] for i in range(dim)])
        a_upper_bound = np.array([self.params_bound[f'a{i+1}'][1] for i in range(dim)])
        c_upper_bound = np.array([self.params_bound[f'{c_prefix}c{i+1}'][1] for i in range(dim)])
        assert np.all(0 < a_lower_bound) and np.all(a_lower_bound < a_upper_bound)
        a = np.random.uniform(0, 1, size=dim)
        a /= np.sum(a)
        a = a_sum * a
        while np.any(a < a_lower_bound) or np.any(a > a_upper_bound):
            a = np.random.uniform(0, 1, size=dim)
            a /= np.sum(a)
            a = a_sum * a

        if 'aL' in self.params_bound:
            aL = [[np.random.uniform(*self.params_bound['aL'])]]
        else:
            aL = []

        params = [np.random.uniform(*self.params_bound['k0'])]  # k0
        for i in range(kappa_dim):
            if i == 1:
                params.append(1/np.random.uniform(*[1/j for j in self.params_bound[f'κ{i+1}'][::-1]]))  # kap2, kap4, ...
            else:
                params.append(np.random.uniform(*self.params_bound[f'κ{i+1}']))  # kap1, kap3,...

        if 'log_ac_bound' in self.apriori_distribution_params:
            c = self.generate_c_from_log_ac(a, a_lower_bound, a_upper_bound, dim)
            if uses_log_c:
                c = np.log(c)
        elif uses_log_c:
            c = np.array([np.random.uniform(*self.params_bound[f'{c_prefix}c{i+1}']) for i in range(dim)])
        else:
            c = self.generate_c_direct(a, a_lower_bound, a_upper_bound, dim)
            if uses_log_c:
                c = np.log(c)

        c = np.clip(c, c_lower_bound, c_upper_bound)
        sorted_ids = np.argsort(c[:self.sort_first])[::-1]

        return np.concatenate([params, c[sorted_ids], c[self.sort_first:], a[sorted_ids], a[self.sort_first:], *aL])

    def generate_c_direct(self, a, a_lower_bound, a_upper_bound, dim):
        log_shift = self.apriori_distribution_params.get('log_shift', 0)
        c = []
        for i in range(dim):
            c.append(
                np.exp(np.random.uniform(*[np.log(j + log_shift) for j in self.params_bound[f'c{i + 1}']])) - log_shift)
        c = np.array(c)
        if (l := self.apriori_distribution_params.get('scale_c', 0)) > 0:
            c *= np.log(1 + (a - a_lower_bound) * l) / np.log(1 + (a_upper_bound - a_lower_bound) * l)
        return c

    def generate_c_from_log_ac(self, a, a_lower_bound, a_upper_bound, dim):
        log_ac_bound = self.apriori_distribution_params['log_ac_bound']
        assert dim == len(log_ac_bound)
        log_ac = np.array([np.random.uniform(*log_ac_bound[i]) for i in range(dim)])
        ac = np.exp(log_ac)
        c = ac / a
        return c

    def make_model(self, theta: np.ndarray | torch.Tensor = None):
        model = self.model_class(
            theta=theta if theta is not None else self.make_random_theta(),
            **self.model_kwargs,
        )
        assert self.labels == model.labels(latex=False), (self.labels, model.labels(latex=False))
        return model


def random_cyclic_plastic_loading(k0: np.ndarray, kap: np.ndarray, a: np.ndarray, c: np.ndarray, epsp: np.ndarray):
    epspc = np.zeros_like(epsp)
    epspc[1:] = np.cumsum(np.abs(epsp[1:] - epsp[:-1]))

    directors = np.append(0, np.sign(epsp[1:] - epsp[:-1]))
    signs = np.sign(directors)
    # Find where the product of consecutive signs is -1 (indicating a sign change)
    reversal_ids = np.append(np.where(signs[:-1] * signs[1:] == -1)[0], -1)  # -1 so that index never reaches it
    return random_cyclic_plastic_loading_engine(k0, kap, a, c, epsp, epspc, directors, reversal_ids)


def random_cyclic_plastic_loading_torch_batch(k0: torch.Tensor, kap: torch.Tensor, a: torch.Tensor, c: torch.Tensor,
                                              epsp: torch.Tensor):
    """
    Each input parameter is a tensor of shape (batch_size, parameter_size), including EPSP. Epsp is expected to have
    the same number of points in each segment so that directors, reversal_ids are identical for all samples in the batch.
    """
    epspc = torch.zeros_like(epsp, device=epsp.device)
    epspc[:, 1:] = torch.cumsum(torch.abs(epsp[:, 1:] - epsp[:, :-1]), dim=1)

    directors = torch.cat((torch.tensor([0], device=epsp.device), torch.sign(
        epsp[0, 1:] - epsp[0, :-1])))  # taking the first sample in the batch - the rest must be the same
    signs = torch.sign(directors)
    # Find where the product of consecutive signs is -1 (indicating a sign change)
    reversal_ids = torch.cat(
        (torch.where(signs[:-1] * signs[1:] == -1)[0],
         torch.tensor([-1], device=epsp.device)))  # -1 so that index never reaches it
    return random_cyclic_plastic_loading_engine_torch(k0, kap, a, c, epsp, epspc, directors, reversal_ids)


@jit(nopython=True)  # this makes the function much faster by compiling it to machine code
def random_cyclic_plastic_loading_engine(k0: np.ndarray, kap: np.ndarray, a: np.ndarray, c: np.ndarray,
                                         epsp: np.ndarray, epspc: np.ndarray,
                                         directors: np.ndarray, reversal_ids: np.ndarray):
    kiso = directors / kap[1] * (1 - (1 - k0[0] * kap[1]) * np.exp(-SQR32 * kap[0] * kap[
        1] * epspc))
    assert len(kiso) == len(epsp)

    alp = np.zeros((len(a), len(epsp)), dtype=np.float64)
    alp_ref = np.zeros(len(a), dtype=np.float64)
    epsp_ref = epsp[0]
    k = 0
    for i, epsp_i in enumerate(epsp):
        # Calculates the absolute value of the new plastic strain increment in the current segment.
        depsp = np.abs(epsp_i - epsp_ref)
        # The evolution of the backstress from the last reference value.
        alp[:, i] = directors[i] * a - (directors[i] * a - alp_ref) * np.exp(-c * depsp)
        # Updates backstress reference values at the beginning of a new segment.
        if i == reversal_ids[k]:
            alp_ref = alp[:, i]
            epsp_ref = epsp_i
            k += 1

    sig = SQR32 * np.sum(alp, axis=0) + kiso  # Overall stress response of the model.
    return sig


def random_cyclic_plastic_loading_engine_torch(k0: torch.Tensor, kap: torch.Tensor, a: torch.Tensor, c: torch.Tensor,
                                               epsp: torch.Tensor, epspc: torch.Tensor,
                                               directors: torch.Tensor, reversal_ids: torch.Tensor):
    kiso = directors.repeat(kap.shape[0], 1) / kap[:, 1:2] * (
                1 - (1 - k0 * kap[:, 1:2]) * torch.exp(-SQR32 * kap[:, 0:1] * kap[:, 1:2] * epspc))
    alp = torch.zeros((*a.shape, epsp.shape[-1]), dtype=kap.dtype, device=epsp.device)
    alp_ref = torch.zeros(*a.shape, dtype=kap.dtype, device=epsp.device)
    epsp_ref = epsp[:, :1]
    last_unprocessed = 0
    for r in reversal_ids:
        if r == -1:
            r = epsp.shape[-1] - 1  # -1 gives the last item's index
            if last_unprocessed >= r:
                break
        depsp = torch.abs(epsp[:, last_unprocessed: r+1] - epsp_ref)
        da = a.unsqueeze(-1) * directors[last_unprocessed: r+1].view(1, 1, -1)
        alp[..., last_unprocessed: r+1] = da - (da - alp_ref.unsqueeze(-1)) * torch.exp(-c.unsqueeze(-1) * depsp.unsqueeze(1))

        alp_ref = alp[..., r]
        epsp_ref = epsp[:, r: r + 1]
        last_unprocessed = r + 1

    sig = SQR32 * torch.sum(alp, dim=1) + kiso
    return sig
