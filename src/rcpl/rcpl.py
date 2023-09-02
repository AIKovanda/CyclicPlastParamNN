import abc

import numpy as np
import torch
from numba import jit

SQR23 = np.sqrt(2. / 3.)
SQR32 = np.sqrt(1.5)


class Experiment:
    PARAM_RANGE = {  # todo
        'k0': (150, 250),
        'kap': ((100, 10000), (1 / 150, 1 / 30)),
        'c': ((np.log(1000), np.log(10000)), (np.log(50), np.log(2000))),
        'a': (0, 350),
    }

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

    def __init__(self, params: dict = None, scaled_params: np.ndarray | torch.Tensor = None, experiment=None,
                 use_torch=False):
        assert (params is None) ^ (scaled_params is None)  # xor - we need exactly one of them to be set
        # params are always stored as a dictionary of unscaled parameters
        self.uses_torch = use_torch
        self.params = params if params is not None else self.unscale_params(scaled_params)
        self.experiment = experiment

    @classmethod
    def scaling_coefficients(cls) -> dict:
        """
        This method needs to be implemented for each dimension of the parameters separately. It returns a dictionary
        in a form of {param_name: (mean, std)}. The mean and std are used to scale and unscale the parameters.
        """
        pass

    @property
    def scaled_params(self) -> np.ndarray | torch.Tensor:
        """
        Returns a vector of scaled parameters. The parameters are scaled using the scaling_coefficients method.
        """
        scaling_coef = self.scaling_coefficients()
        values = [
            self.scale(self.params['k0'], scaling_coef['k0']),
            self.scale(self.params['kap'][0], scaling_coef['kap'][0]),
            self.scale(1 / self.params['kap'][1], scaling_coef['kap'][1]),
        ]
        values += [self.scale(np.log(c), coef) for c, coef in zip(self.params['c'], scaling_coef['c'])]
        values += [self.scale(a, scaling_coef['a']) for a in self.params['a']]
        return np.array(values) if not self.uses_torch else torch.Tensor(values)

    def unscale_params(self, scaled_params: np.ndarray | torch.Tensor) -> dict:
        """
        This is an inverse method to scaled_params. It returns a dictionary of parameters in unscaled form.
        """
        scaling_coef = self.scaling_coefficients()
        c_len = len(scaling_coef['c'])
        assert len(scaled_params) == 3 + 2 * c_len
        if self.uses_torch:
            return {
                'k0': self.unscale(scaled_params[0], scaling_coef['k0']),
                'kap': [self.unscale(scaled_params[1], scaling_coef['kap'][0]),
                        1 / self.unscale(scaled_params[2], scaling_coef['kap'][1])],
                'c': torch.exp(torch.tensor(
                    [self.unscale(scaled_params[i + 3], coef) for i, coef in enumerate(scaling_coef['c'])])),
                'a': torch.tensor([self.unscale(a, scaling_coef['a']) for a in scaled_params[3 + c_len:]]),
            }
        return {
            'k0': self.unscale(scaled_params[0], scaling_coef['k0']),
            'kap': np.array([self.unscale(scaled_params[1], scaling_coef['kap'][0]),
                             1 / self.unscale(scaled_params[2], scaling_coef['kap'][1])]),
            'c': np.exp(np.array(
                [self.unscale(scaled_params[i + 3], coef) for i, coef in enumerate(scaling_coef['c'])])),
            'a': np.array([self.unscale(a, scaling_coef['a']) for a in scaled_params[3 + c_len:]]),
        }

    @staticmethod
    def _uniform_e_var(interval: tuple[float, float]) -> tuple[float, float]:
        """
        This helping method returns a tuple of mean and standard deviation of a uniform distribution given its interval.
        """
        a, b = interval
        assert b > a
        return (a + b) / 2, (b - a) / np.sqrt(12)

    @staticmethod
    def scale(num, e_var: tuple[float, float]) -> float:
        e, var = e_var
        return (num - e) / var

    @staticmethod
    def unscale(num, e_var: tuple[float, float]) -> float:
        e, var = e_var
        return max(1e-9, num * var + e)

    @classmethod
    def generate_params(cls, experiment=None, use_torch=False, **kwargs):
        """
        Generates random parameters for the cyclic plastic loading model. Returns an instance of this class.
        """
        scaling_coef = cls.scaling_coefficients()
        c_len = len(scaling_coef['c'])
        if use_torch:
            a = torch.rand(c_len)
            a /= torch.sum(a)
        else:
            a = np.random.uniform(0, 1, size=c_len)
            a /= np.sum(a)
        a = np.random.uniform(150, 350) * a

        if use_torch:
            params = {
                'k0': torch.Tensor([np.random.uniform(10, 250)]),
                'kap': torch.Tensor([np.random.uniform(100, 10000), 1 / np.random.uniform(30, 200)]),
                'c': torch.from_numpy(np.sort([np.exp(np.random.uniform(np.log(1000), np.log(10000))),
                                               *np.exp(np.random.uniform(np.log(1e-9), np.log(2000), size=c_len - 1))])[
                                      ::-1].copy()),
                'a': a,
            }
        else:
            params = {
                'k0': np.random.uniform(15, 250),
                'kap': np.array([np.random.uniform(100, 10000), 1 / np.random.uniform(30, 150)]),
                'c': np.sort([np.exp(np.random.uniform(np.log(1000), np.log(10000))),
                              *np.exp(np.random.uniform(np.log(50), np.log(2000), size=c_len - 1))])[::-1],
                'a': a,
            }
        return cls(
            params=params,
            experiment=experiment,
            use_torch=use_torch,
        )


class RandomCyclicPlasticLoadingParams1D(RandomCyclicPlasticLoadingParams):

    @classmethod
    def scaling_coefficients(cls) -> dict:
        return {
            'k0': cls._uniform_e_var((150, 250)),
            'kap': (cls._uniform_e_var((100, 10000)), cls._uniform_e_var((30, 150))),
            'c': (cls._uniform_e_var((np.log(1000), np.log(10000))),),
            'a': cls._uniform_e_var((250, 350)),
        }


class RandomCyclicPlasticLoadingParams2D(RandomCyclicPlasticLoadingParams):

    @classmethod
    def scaling_coefficients(cls) -> dict:
        return {
            'k0': cls._uniform_e_var((150, 250)),
            'kap': (cls._uniform_e_var((100, 10000)), cls._uniform_e_var((30, 150))),
            'c': ((8.06, 0.65), (5.75, 1.05)),
            'a': (150, 73.5),
        }


class RandomCyclicPlasticLoadingParams3D(RandomCyclicPlasticLoadingParams):

    @classmethod
    def scaling_coefficients(cls) -> dict:
        return {
            'k0': cls._uniform_e_var((150, 250)),
            'kap': (cls._uniform_e_var((100, 10000)), cls._uniform_e_var((30, 150))),
            'c': ((8.06, 0.65), (6.36, 0.86), (5.15, 0.87)),
            'a': (100, 54.9),
        }


class RandomCyclicPlasticLoadingParams4D(RandomCyclicPlasticLoadingParams):

    @classmethod
    def scaling_coefficients(cls) -> dict:
        return {
            'k0': cls._uniform_e_var((150, 250)),
            'kap': (cls._uniform_e_var((100, 10000)), cls._uniform_e_var((30, 150))),
            'c': (
                (8.07521718, 0.64220986), (6.66012441, 0.70230589), (5.75443793, 0.82282555), (4.83443022, 0.71284407)),
            'a': (75, 42.8),
        }


def random_cyclic_plastic_loading(k0: float, kap: np.ndarray, a: np.ndarray, c: np.ndarray, epsp: np.ndarray):
    epspc = np.zeros_like(epsp)
    epspc[1:] = np.cumsum(np.abs(epsp[1:] - epsp[:-1]))

    directors = np.append(0, np.sign(epsp[1:] - epsp[:-1]))
    signs = np.sign(directors)
    # Find where the product of consecutive signs is -1 (indicating a sign change)
    reversal_ids = np.append(np.where(signs[:-1] * signs[1:] == -1)[0], -1)  # -1 so that index never reaches it
    return random_cyclic_plastic_loading_engine(k0, kap, a, c, epsp, epspc, directors, reversal_ids)


@jit(nopython=True)  # this makes the function much faster by compiling it to machine code
def random_cyclic_plastic_loading_engine(k0: float, kap: np.ndarray, a: np.ndarray, c: np.ndarray,
                                         epsp: np.ndarray, epspc: np.ndarray,
                                         directors: np.ndarray, reversal_ids: np.ndarray):
    kiso = directors / kap[1] * (1 - (1 - k0 * kap[1]) * np.exp(-SQR32 * kap[0] * kap[
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
def rt(theta: torch.Tensor, epsp: torch.Tensor) -> torch.Tensor:
    return random_cyclic_plastic_loading_t(
        epsp=epsp,
        kap=theta[:3],
        c=theta[3:7],
        a=theta[7:11],
    )


def get_random_pseudo_experiment(dim, kappa_dim, experiment: Experiment, scaled_params=None):
    if scaled_params is not None:
        to_eval = f"RandomCyclicPlasticLoadingParams{dim}D"
    else:
        to_eval = f"RandomCyclicPlasticLoadingParams{dim}D.generate_params"
    model_params = eval(to_eval)(
        experiment=experiment,
        scaled_params=scaled_params,
    )
    sig = random_cyclic_plastic_loading(**model_params.params, epsp=experiment.epsp)
    if sig.max() > 5000:
        print(f'Parameters: {model_params.params}. Max of sig is {sig.max()}.')
    return sig, model_params
