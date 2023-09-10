import abc

import numpy as np
import torch
from numba import jit

SQR23 = np.sqrt(2./3.)
SQR32 = np.sqrt(1.5)


class Experiment:
    def __init__(self, epsp: list | np.ndarray | torch.Tensor, **meta):
        assert epsp is not None
        if isinstance(epsp, list):
            epsp = np.array(epsp)
        self.epsp = epsp
        self.meta = meta

    @classmethod
    def random_experiment(cls, uniform_params: tuple[int, int, int], points_per_segment: int, method="geometric",
                          first_last_ratio=20, **meta):
        epsp_r = np.append(0, np.random.uniform(*uniform_params))
        epsp_r[::2] *= -1
        if method == "linear":
            epsp = [np.array([0])]
            for segment_id in range(uniform_params[-1]):
                epsp.append(np.linspace(epsp_r[segment_id], epsp_r[segment_id + 1], points_per_segment+1)[1:])
            epsp = np.concatenate(epsp)

        elif method == "geometric":
            r = first_last_ratio ** (1 / (points_per_segment - 2))  # Ratio between steps.
            geospace = np.array([r ** i for i in range(1, points_per_segment)])
            geospace /= np.sum(geospace)
            geospace = np.cumsum(geospace)
            epsp = [0]
            for segment_id in range(uniform_params[-1]):
                epsp.extend(
                    [epsp_r[segment_id + 1] * geo_val + (1 - geo_val) * epsp_r[segment_id] for geo_val in geospace])
            epsp = np.array(epsp)
        else:
            raise ValueError(f"Unknown method {method}.")
        return cls(epsp=epsp, **meta)

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


def random_cyclic_plastic_loading_torch_batch(k0: torch.Tensor, kap: torch.Tensor, a: torch.Tensor, c: torch.Tensor, epsp: torch.Tensor):
    """
    Each input parameter is a tensor of shape (batch_size, parameter_size), including EPSP. Epsp is expected to have
    the same number of points in each segment so that directors, reversal_ids are identical for all samples in the batch.
    """
    epspc = torch.zeros_like(epsp, device=epsp.device)
    epspc[:, 1:] = torch.cumsum(torch.abs(epsp[:, 1:] - epsp[:, :-1]), dim=1)

    directors = torch.cat((torch.tensor([0], device=epsp.device), torch.sign(epsp[0, 1:] - epsp[0, :-1])))  # taking the first sample in the batch - the rest must be the same
    signs = torch.sign(directors)
    # Find where the product of consecutive signs is -1 (indicating a sign change)
    reversal_ids = torch.cat(
        (torch.where(signs[:-1] * signs[1:] == -1)[0],
         torch.tensor([-1], device=epsp.device)))  # -1 so that index never reaches it
    return random_cyclic_plastic_loading_engine_t(k0, kap, a, c, epsp, epspc, directors, reversal_ids)


def random_cyclic_plastic_loading_engine_t(k0: torch.Tensor, kap: torch.Tensor, a: torch.Tensor, c: torch.Tensor,
                                           epsp: torch.Tensor, epspc: torch.Tensor,
                                           directors: torch.Tensor, reversal_ids: torch.Tensor):
    kiso = directors.repeat(kap.shape[0], 1) / kap[:, 1:2] * (1 - (1 - k0 * kap[:, 1:2]) * torch.exp(-SQR32 * kap[:, 0:1] * kap[:, 1:2] * epspc))
    alp = torch.zeros((*a.shape, epsp.shape[-1]), dtype=kap.dtype, device=epsp.device)
    alp_ref = torch.zeros(*a.shape, dtype=kap.dtype, device=epsp.device)
    epsp_ref = epsp[:, :1]
    k = 0
    for i in range(epsp.shape[-1]):
        depsp = torch.abs(epsp[:, i: i+1] - epsp_ref)
        alp[..., i] = directors[i] * a - (directors[i] * a - alp_ref) * torch.exp(-c * depsp)
        if i == reversal_ids[k]:  # Use .item() to extract the value from the tensor
            alp_ref = alp[..., i]
            epsp_ref = epsp[:, i: i+1]
            k += 1

    sig = SQR32 * torch.sum(alp, dim=1) + kiso
    return sig


@torch.compile
def rcpl_torch_one(theta: torch.Tensor, epsp: torch.Tensor, dim, kappa_dim) -> torch.Tensor:
    return random_cyclic_plastic_loading_torch_batch(
        epsp=torch.unsqueeze(epsp, 0),
        k0=torch.unsqueeze(theta[:1], 0),
        kap=torch.unsqueeze(theta[1:1 + kappa_dim], 0),
        c=torch.unsqueeze(theta[1 + kappa_dim:1 + kappa_dim + dim], 0),
        a=torch.unsqueeze(theta[1 + kappa_dim + dim:], 0),
    )[0]


#@torch.compile
def rcpl_torch_batch(theta: torch.Tensor, epsp: torch.Tensor, dim, kappa_dim) -> torch.Tensor:
    return random_cyclic_plastic_loading_torch_batch(
        epsp=epsp,
        k0=theta[:, :1],
        kap=theta[:, 1:1 + kappa_dim],
        c=theta[:, 1 + kappa_dim:1 + kappa_dim + dim],
        a=theta[:, 1 + kappa_dim + dim:],
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
