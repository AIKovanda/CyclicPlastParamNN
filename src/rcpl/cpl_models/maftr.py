import numpy as np
import torch
from numba import jit
from taskchain.parameter import AutoParameterObject

from rcpl.cpl_models import CPLModelFactory
from rcpl.cpl_models.maf import MAFModel

SQR32 = np.sqrt(1.5)


class MAFTrCPLModelFactory(AutoParameterObject, CPLModelFactory):

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
            1 / np.random.uniform(*[1 / j for j in self.params_bound['κ2'][::-1]]),  # kap2
        ]
        c = []
        for i in range(dim):
            c.append(np.exp(np.random.uniform(*[np.log(j) for j in self.params_bound[f'c{i + 1}']])))

        return np.concatenate([params, c, a])

    def make_model(self, theta: np.ndarray = None):
        model = MAFTr(
            theta=theta if theta is not None else self.make_random_theta(),
            **self.model_kwargs,
        )
        assert self.labels == model.labels(latex=False), (self.labels, model.labels(latex=False))
        return model


class MAFTr(MAFModel):
    model_name = 'MAFTr'

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


@jit(nopython=True)  # this makes the function much faster by compiling it to machine code
def random_cyclic_plastic_loading_engine(k0: np.ndarray, kap: np.ndarray, a: np.ndarray, c: np.ndarray,
                                         epsp: np.ndarray, epspc: np.ndarray,
                                         directors: np.ndarray, reversal_ids: np.ndarray):
    kiso = directors / kap[1] * (1 - (1 - k0[0] * kap[1]) * np.exp(-SQR32 * kap[0] * kap[
        1] * epspc))  # Vyvoj kiso jako funkce epspc, rovnou orientovane ve smeru narustajici plasticke deformace.
    #    kiso = kiso + directors/kap[4]*(1-np.exp(-SQR32 * kap[3] * kap[4] * epspc))  # Odkomentuj k aktivaci druhe funkce isotropniho zpevneni
    #    kiso = kiso + directors/kap[6]*(1-np.exp(-SQR32 * kap[5] * kap[6] * epspc))  # Odkomentuj k aktivaci treti funkce isotropniho zpevneni
    assert len(kiso) == len(epsp)

    alp = np.zeros((len(a) - 1, len(epsp)), dtype=np.float64)
    alp_ref = np.zeros(len(a), dtype=np.float64)
    epsp_ref = epsp[0]
    k = 0
    depsp_thr = 0
    for i, epsp_i in enumerate(epsp):
        # Spocita absolutni hodnotu noveho prirustku plasticke deformace v aktualnim segmentu
        depsp = np.abs(epsp_i - epsp_ref)
        # Vyvoj backstressu od posledni referencni hodnoty.
        alp[0:3, i] = directors[i] * a[0:3] - (directors[i] * a[0:3] - alp_ref[0:3]) * np.exp(-c[0:3] * depsp)
        # Prepisu posledni segment backstressu
        alp[3, i] = alp_ref[3]
        if directors[i] * alp[3, i] < -a[4]:  # abar
            depsp_thr = -1. / c[3] * np.log(a[3] / (a[3] - a[4] - directors[i] * alp[3, i]))
            alp[:, 3] = directors[i] * (a[3] - a[4]) - (directors * (a[3] - a[4]) - alp[3]) * np.exp(
                -c[3] * np.min(depsp, depsp_thr))
            depsp = np.max(0, depsp - depsp_thr)
        if depsp > 0 and (depsp_thr > 0 or directors[i] * alp[3, i] <= a[4]):
            depsp_thr = (a[4] - directors[i] * alp[3, i]) / c[3] / a[3]
            alp[3, i] = alp[3, i] + directors[i] * c[3] * a[3] * min(depsp, depsp_thr)
            depsp = max(0, depsp - depsp_thr)
        if depsp > 0:
            alp[3] = directors[i] * (a[3] + a[4]) - (directors[i] * (a[3] + a[4]) - alp[3]) * np.exp(-c[3] * depsp)
        # Aktualizuje referencni hodnoty backstressu na zacatku noveho segmentu
        if i == reversal_ids[k]:
            alp_ref = alp[:, i]
            epsp_ref = epsp_i
            k += 1

    sig = SQR32 * np.sum(alp, axis=0) + kiso  # Celkova napetova odezva modelu.
    return sig
