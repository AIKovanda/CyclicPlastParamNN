from abc import ABC
from functools import cache

import numpy as np
import torch
from numba import jit
from taskchain.parameter import AutoParameterObject

from rcpl.cpl_models import CPLModel, CPLModelFactory

SQR32 = np.sqrt(1.5)


class MAFCPLModelFactory(AutoParameterObject, CPLModelFactory):

    def make_random_theta(self):
        """
        Generates random parameters for the MAF model.
        """
        dim = self.model_kwargs['dim']
        kappa_dim = self.model_kwargs['kappa_dim']
        assert kappa_dim == 2

        a = np.random.uniform(0, 1, size=dim)
        a /= np.sum(a)
        a = np.random.uniform(150, 350) * a

        params = [
            np.random.uniform(*self.params_bound['k0']),  # k0
            np.random.uniform(*self.params_bound['κ1']),  # kap1
            1/np.random.uniform(*[1/j for j in self.params_bound['κ2'][::-1]]),  # kap2
        ]
        c = []
        for i in range(dim):
            c.append(np.exp(np.random.uniform(*[np.log(j) for j in self.params_bound[f'c{i+1}']])))

        return np.concatenate([params, np.sort(c)[::-1], a])

    def make_model(self, theta: np.ndarray = None):
        model = MAF(
            theta=theta if theta is not None else self.make_random_theta(),
            **self.model_kwargs,
        )
        assert self.labels == model.labels(latex=False), (self.labels, model.labels(latex=False))
        return model


class MAFModel(CPLModel, ABC):
    model_name = None

    def __init__(self, theta: np.ndarray | torch.Tensor = None, dim=4, kappa_dim=2):
        assert self.model_name in ['MAF', 'MAFTr']
        self.dim = dim
        self.kappa_dim = kappa_dim
        super().__init__(theta=theta)

    @property
    def theta_len(self):
        return 1 + self.kappa_dim + self.dim * 2 + (1 if self.model_name == 'MAFTr' else 0)

    def _get_labels(self, latex=False):
        label_key = 'latex_label' if latex else 'label'
        uses_params = {
            'k0': {'dim': 1, 'label': 'k0', 'latex_label': 'k_0'},
            'kap': {'dim': self.kappa_dim, 'label': 'κ', 'latex_label': '\\kappa'},
            'c': {'dim': self.dim, 'label': 'c', 'latex_label': 'c'},
            'a': {'dim': self.dim, 'label': 'a', 'latex_label': 'a'},
        }
        if self.model_name == 'MAFTr':
            uses_params['aL'] = {'dim': 1, 'label': 'aL', 'latex_label': '\\overline{a}'}

        labels = []
        for param_name, param_info in uses_params.items():
            if param_info['dim'] == 1:
                labels.append(param_info[label_key])
            else:
                for i in range(param_info['dim']):
                    labels.append(f'{param_info[label_key]}{i + 1}')
        return labels


class MAF(MAFModel):
    model_name = 'MAF'

    def _predict_stress(self, epsp: np.ndarray) -> np.ndarray:
        return random_cyclic_plastic_loading(
            k0=self.theta[:1],
            kap=self.theta[1:1 + self.kappa_dim],
            c=self.theta[1 + self.kappa_dim:1 + self.kappa_dim + self.dim],
            a=self.theta[1 + self.kappa_dim + self.dim:],
            epsp=epsp,
        )

    def _predict_stress_torch(self, epsp: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        return random_cyclic_plastic_loading_torch_batch(
            epsp=torch.unsqueeze(epsp, 0),
            k0=torch.unsqueeze(self.theta[:1], 0),
            kap=torch.unsqueeze(self.theta[1:1 + self.kappa_dim], 0),
            c=torch.unsqueeze(self.theta[1 + self.kappa_dim:1 + self.kappa_dim + self.dim], 0),
            a=torch.unsqueeze(self.theta[1 + self.kappa_dim + self.dim:], 0),
        )[0]

    def _predict_stress_torch_batch(self, epsp: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        return random_cyclic_plastic_loading_torch_batch(
            epsp=epsp,
            k0=self.theta[:, :1],
            kap=self.theta[:, 1:1 + self.kappa_dim],
            c=self.theta[:, 1 + self.kappa_dim:1 + self.kappa_dim + self.dim],
            a=self.theta[:, 1 + self.kappa_dim + self.dim:],
        )


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
        1] * epspc))  # Vyvoj kiso jako funkce epspc, rovnou orientovane ve smeru narustajici plasticke deformace.
    #    kiso = kiso + directors/kap[4]*(1-np.exp(-SQR32 * kap[3] * kap[4] * epspc))  # Odkomentuj k aktivaci druhe funkce isotropniho zpevneni
    #    kiso = kiso + directors/kap[6]*(1-np.exp(-SQR32 * kap[5] * kap[6] * epspc))  # Odkomentuj k aktivaci treti funkce isotropniho zpevneni
    assert len(kiso) == len(epsp)

    alp = np.zeros((len(a), len(epsp)), dtype=np.float64)
    alp_ref = np.zeros(len(a), dtype=np.float64)
    epsp_ref = epsp[0]
    k = 0
    for i, epsp_i in enumerate(epsp):
        # Spocita absolutni hodnotu noveho prirustku plasticke deformace v aktualnim segmentu
        depsp = np.abs(epsp_i - epsp_ref)
        # Vyvoj backstressu od posledni referencni hodnoty.
        alp[:, i] = directors[i] * a - (directors[i] * a - alp_ref) * np.exp(-c * depsp)
        # Aktualizuje referencni hodnoty backstressu na zacatku noveho segmentu
        if i == reversal_ids[k]:
            alp_ref = alp[:, i]
            epsp_ref = epsp_i
            k += 1

    sig = SQR32 * np.sum(alp, axis=0) + kiso  # Celkova napetova odezva modelu.
    return sig


def random_cyclic_plastic_loading_engine_torch(k0: torch.Tensor, kap: torch.Tensor, a: torch.Tensor, c: torch.Tensor,
                                               epsp: torch.Tensor, epspc: torch.Tensor,
                                               directors: torch.Tensor, reversal_ids: torch.Tensor):
    kiso = directors.repeat(kap.shape[0], 1) / kap[:, 1:2] * (
                1 - (1 - k0 * kap[:, 1:2]) * torch.exp(-SQR32 * kap[:, 0:1] * kap[:, 1:2] * epspc))
    alp = torch.zeros((*a.shape, epsp.shape[-1]), dtype=kap.dtype, device=epsp.device)
    alp_ref = torch.zeros(*a.shape, dtype=kap.dtype, device=epsp.device)
    epsp_ref = epsp[:, :1]
    k = 0
    for i in range(epsp.shape[-1]):
        depsp = torch.abs(epsp[:, i: i + 1] - epsp_ref)
        alp[..., i] = directors[i] * a - (directors[i] * a - alp_ref) * torch.exp(-c * depsp)
        if i == reversal_ids[k]:  # Use .item() to extract the value from the tensor
            alp_ref = alp[..., i]
            epsp_ref = epsp[:, i: i + 1]
            k += 1

    sig = SQR32 * torch.sum(alp, dim=1) + kiso
    return sig


#
#
# def get_random_pseudo_experiment(dim, kappa_dim, experiment: Experiment, vector_params: np.ndarray = None):
#     if vector_params is None:
#         model_params = RandomCyclicPlasticLoadingParams.generate_params(
#             experiment=experiment,
#             dim=dim,
#             kappa_dim=kappa_dim,
#         )
#     else:
#         model_params = RandomCyclicPlasticLoadingParams(
#             params=vector_params,
#             experiment=experiment,
#             dim=dim,
#             kappa_dim=kappa_dim,
#         )
#     sig = random_cyclic_plastic_loading(**model_params.params, epsp=experiment.epsp)
#     # if sig.max() > 5000:
#     #     print(f'Parameters: {model_params.params}. Max of sig is {sig.max()}.')
#     return sig, model_params
