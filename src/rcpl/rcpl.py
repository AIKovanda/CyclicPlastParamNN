import abc

import numpy as np
import torch
from numba import jit

SQR23 = np.sqrt(2. / 3.)
SQR32 = np.sqrt(1.5)


class Experiment:
    def __init__(self, epsp: list | np.ndarray | torch.Tensor = None, **meta):
        assert epsp is not None
        if isinstance(epsp, list):
            epsp = np.array(epsp)
        self._epsp = epsp
        self.meta = meta

    @property
    def epsp(self):
        if self._epsp is None:
            depspc_r = np.random.uniform(0.0005, 0.005, self.meta['depspc_r_number'])
            self._epsp = self.depsp_r2epsp(depspc_r)
            raise NotImplementedError
        return self._epsp

    @staticmethod
    def depsp_r2epsp(depsp_r: np.ndarray) -> np.ndarray:
        epsp_r = np.append(0, np.cumsum(depsp_r))
        epsp = []
        for segment_id in range(12):
            epsp.append(np.linspace(epsp_r[segment_id], epsp_r[segment_id + 1], 21)[:-1])  # todo: geometric
        epsp = np.concatenate(epsp)
        return epsp

    def __len__(self):
        return len(self.epsp)


class RandomCyclicPlasticLoadingParams(abc.ABC):
    """
    This class is used to store parameters of the random cyclic plastic loading model. It is used to generate
    random parameters for cyclic plastic loading and to scale and unscale parameters (normalization).
    """

    def __init__(self, params: dict | np.ndarray, dim: int, kappa_dim=2, experiment=None, use_torch=False):
        self.uses_torch = use_torch
        self.experiment = experiment
        self.dim = dim
        self.kappa_dim = kappa_dim
        self.params = params if isinstance(params, dict) else self.vector_params_to_dict(params)

    @property
    def param_labels(self):
        labels = ['k0']
        for i in range(self.kappa_dim):
            labels.append(f'κ{i+1}')
        for i in range(self.dim):
            labels.append(f'c{i+1}')
        for i in range(self.dim):
            labels.append(f'a{i+1}')
        return labels

    @property
    def latex_param_labels(self):
        labels = ['k_0']
        for i in range(self.kappa_dim):
            labels.append(f'\\kappa_{i+1}')
        for i in range(self.dim):
            labels.append(f'c_{i+1}')
        for i in range(self.dim):
            labels.append(f'a_{i+1}')
        return labels
        
    @property
    def vector_params(self):
        return np.concatenate([self.params['k0'], self.params['kap'], self.params['c'], self.params['a']])

    def vector_params_to_dict(self, params):
        return {
            'k0': params[:1],
            'kap': params[1:1 + self.kappa_dim],
            'c': params[1 + self.kappa_dim:1 + self.kappa_dim + self.dim],
            'a': params[1 + self.kappa_dim + self.dim:],
        }

    @classmethod
    def generate_params(cls, dim: int, kappa_dim: int = 2, experiment=None, use_torch=False):
        """
        Generates random parameters for the cyclic plastic loading model. Returns an instance of this class.
        """
        if kappa_dim != 2:
            raise NotImplementedError

        a = np.random.uniform(0, 1, size=dim)
        a /= np.sum(a)
        a = np.random.uniform(150, 350) * a

        params = {
            'k0': np.random.uniform(15, 250, size=1),
            'kap': np.array([np.random.uniform(100, 10000), 1 / np.random.uniform(30, 150)]),
            'c': np.sort([np.exp(np.random.uniform(np.log(1000), np.log(10000))),
                          *np.exp(np.random.uniform(np.log(50), np.log(2000), size=dim - 1))])[::-1],
            'a': a,
        }
        if use_torch:
            params = {key: torch.from_numpy(value.copy()) for key, value in params.items()}

        return cls(
            params=params,
            experiment=experiment,
            use_torch=use_torch,
            dim=dim,
            kappa_dim=kappa_dim,
        )


def random_cyclic_plastic_loading(k0: np.ndarray, kap: np.ndarray, a: np.ndarray, c: np.ndarray, epsp: np.ndarray):
    epspc = np.zeros_like(epsp)
    epspc[1:] = np.cumsum(np.abs(epsp[1:] - epsp[:-1]))

    directors = np.append(0, np.sign(epsp[1:] - epsp[:-1]))
    signs = np.sign(directors)
    # Find where the product of consecutive signs is -1 (indicating a sign change)
    reversal_ids = np.append(np.where(signs[:-1] * signs[1:] == -1)[0], -1)  # -1 so that index never reaches it
    return random_cyclic_plastic_loading_engine(k0, kap, a, c, epsp, epspc, directors, reversal_ids)


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


def random_cyclic_plastic_loading_t(k0: float, kap: torch.Tensor, a: torch.Tensor, c: torch.Tensor, epsp: torch.Tensor):
    epspc = torch.zeros_like(epsp, device=epsp.device)
    epspc[1:] = torch.cumsum(torch.abs(epsp[1:] - epsp[:-1]), dim=0)

    directors = torch.cat((torch.tensor([0], device=epsp.device), torch.sign(epsp[1:] - epsp[:-1])))
    signs = torch.sign(directors)
    # Find where the product of consecutive signs is -1 (indicating a sign change)
    product_of_consecutive_signs = signs[:-1] * signs[1:]
    reversal_ids = torch.cat(
        (torch.where(product_of_consecutive_signs == -1)[0],
         torch.tensor([-1], device=epsp.device)))  # -1 so that index never reaches it
    return random_cyclic_plastic_loading_engine_t(k0, kap, a, c, epsp, epspc, directors, reversal_ids)


def random_cyclic_plastic_loading_engine_t(k0: float, kap: torch.Tensor, a: torch.Tensor, c: torch.Tensor,
                                           epsp: torch.Tensor, epspc: torch.Tensor,
                                           directors: torch.Tensor, reversal_ids: torch.Tensor):
    kiso = directors / kap[1] * (1 - (1 - k0 * kap[1]) * torch.exp(-SQR32 * kap[0] * kap[1] * epspc))
    assert len(kiso) == len(epsp)
    alp = torch.zeros((len(a), len(epsp)), dtype=torch.float64, device=epsp.device)
    alp_ref = torch.zeros(len(a), dtype=torch.float64, device=epsp.device)
    epsp_ref = epsp[0]
    k = 0
    for i, epsp_i in enumerate(epsp):
        depsp = torch.abs(epsp_i - epsp_ref)
        alp[:, i] = directors[i] * a - (directors[i] * a - alp_ref) * torch.exp(-c * depsp)
        if i == reversal_ids[k]:  # Use .item() to extract the value from the tensor
            alp_ref = alp[:, i]
            epsp_ref = epsp_i
            k += 1

    sig = SQR32 * torch.sum(alp, dim=0) + kiso
    return sig


@torch.compile
def rt(theta: torch.Tensor, epsp: torch.Tensor) -> torch.Tensor:  # todo
    return random_cyclic_plastic_loading_t(
        epsp=epsp,
        kap=theta[:3],
        c=theta[3:7],
        a=theta[7:11],
    )


def get_random_pseudo_experiment(dim, kappa_dim, experiment: Experiment, vector_params: np.ndarray = None):
    if vector_params is None:
        model_params = RandomCyclicPlasticLoadingParams.generate_params(
            experiment=experiment,
            dim=dim,
            kappa_dim=kappa_dim,
        )
    else:
        model_params = RandomCyclicPlasticLoadingParams(
            params=vector_params,
            experiment=experiment,
            dim=dim,
            kappa_dim=kappa_dim,
        )
    sig = random_cyclic_plastic_loading(**model_params.params, epsp=experiment.epsp)
    if sig.max() > 5000:
        print(f'Parameters: {model_params.params}. Max of sig is {sig.max()}.')
    return sig, model_params
