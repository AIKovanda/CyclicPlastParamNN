import numpy as np
import torch
from numba import jit

from rcpl.cpl_models.maf import MAFModel, MAFCPLModelFactory

SQR32 = np.sqrt(1.5)


class MAFTr(MAFModel):
    model_name = 'MAFTr'

    def _predict_stress(self, signal: np.ndarray) -> np.ndarray:
        assert len(set(signal.shape) - {1}) == 1
        return random_cyclic_plastic_loading(
            k0=self.theta[:1],
            kap=self.theta[1:1 + self.kappa_dim],
            c=self.theta[1 + self.kappa_dim:1 + self.kappa_dim + self.dim],
            a=self.theta[1 + self.kappa_dim + self.dim:],  # includes abar
            epsp=signal.flatten(),
        )

    def _predict_stress_torch(self, signal: torch.Tensor) -> torch.Tensor:
        assert len(set(signal.shape) - {1}) == 1
        return random_cyclic_plastic_loading_torch_batch(
            epsp=torch.unsqueeze(signal, 0),
            k0=torch.unsqueeze(self.theta[:1], 0),
            kap=torch.unsqueeze(self.theta[1:1 + self.kappa_dim], 0),
            c=torch.unsqueeze(self.theta[1 + self.kappa_dim:1 + self.kappa_dim + self.dim], 0),
            a=torch.unsqueeze(self.theta[1 + self.kappa_dim + self.dim:], 0),
        )[0]

    def _predict_stress_torch_batch(self, signal: torch.Tensor) -> torch.Tensor:
        assert signal.ndim == 3
        assert signal.shape[1] == 1
        return random_cyclic_plastic_loading_torch_batch(
            epsp=signal[:, 0, :],
            k0=self.theta[:, :1],
            kap=self.theta[:, 1:1 + self.kappa_dim],
            c=self.theta[:, 1 + self.kappa_dim:1 + self.kappa_dim + self.dim],
            a=self.theta[:, 1 + self.kappa_dim + self.dim:],
        )


class MAFTrCPLModelFactory(MAFCPLModelFactory):
    sort_first = 4
    model_class = MAFTr


def random_cyclic_plastic_loading(k0: np.ndarray, kap: np.ndarray, a: np.ndarray, c: np.ndarray, epsp: np.ndarray) -> np.ndarray:
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
    a = a * np.sqrt(2/3)
    kiso = directors / kap[1] * (1 - (1 - k0[0] * kap[1]) * np.exp(-SQR32 * kap[0] * kap[1] * epspc))  # Vyvoj kiso jako funkce epspc, rovnou orientovane ve smeru narustajici plasticke deformace.

    for k_dim in range(2, kap.shape[0], 2):
        kiso = kiso + directors/kap[k_dim+1]*(1-np.exp(-SQR32 * kap[k_dim] * kap[k_dim+1] * epspc))  # Odkomentuj k aktivaci druhe funkce isotropniho zpevneni

    assert len(kiso) == len(epsp)

    alp = np.zeros((len(a) - 1, len(epsp)), dtype=np.float64)
    alp_ref = np.zeros(len(a)-1, dtype=np.float64)
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
            alp[3, i] = directors[i] * (a[3] - a[4]) - (directors[i] * (a[3] - a[4]) - alp[3, i]) * np.exp(-c[3] * min(depsp, depsp_thr))
            depsp = max(0, depsp - depsp_thr)
        if depsp > 0 and (depsp_thr > 0 or directors[i] * alp[3, i] <= a[4]):
            depsp_thr = (a[4] - directors[i] * alp[3, i]) / c[3] / a[3]
            alp[3, i] = alp[3, i] + directors[i] * c[3] * a[3] * min(depsp, depsp_thr)
            depsp = max(0, depsp - depsp_thr)
        if depsp > 0:
            alp[3, i] = directors[i] * (a[3] + a[4]) - (directors[i] * (a[3] + a[4]) - alp[3, i]) * np.exp(-c[3] * depsp)
        # Aktualizuje referencni hodnoty backstressu na zacatku noveho segmentu
        if i == reversal_ids[k]:
            alp_ref = alp[:, i]
            epsp_ref = epsp_i
            k += 1

    np.save('alp.npy', alp)
    sig = SQR32 * np.sum(alp, axis=0) + kiso  # Celkova napetova odezva modelu.
    return sig


def random_cyclic_plastic_loading_engine_torch(k0: torch.Tensor, kap: torch.Tensor, a: torch.Tensor, c: torch.Tensor,
                                               epsp: torch.Tensor, epspc: torch.Tensor,
                                               directors: torch.Tensor, reversal_ids: torch.Tensor):
    a = a * np.sqrt(2/3)

    rep_dir = directors.repeat(kap.shape[0], 1)
    kiso = rep_dir / kap[:, 1:2] * (1 - (1 - k0 * kap[:, 1:2]) * torch.exp(-SQR32 * kap[:, 0:1] * kap[:, 1:2] * epspc))

    for k_dim in range(2, kap.shape[1], 2):
        kiso = kiso + rep_dir/kap[:, k_dim+1: k_dim+2]*(1-np.exp(-SQR32 * kap[:, k_dim: k_dim+1] * kap[:, k_dim+1: k_dim+2] * epspc))

    alp = torch.zeros((*[a.shape[0], a.shape[1]-1], epsp.shape[-1]), dtype=kap.dtype, device=epsp.device)
    alp_ref = torch.zeros([a.shape[0], a.shape[1]-1], dtype=kap.dtype, device=epsp.device)
    epsp_ref = epsp[:, :1]
    last_unprocessed = 0
    for r in reversal_ids:
        if r == -1:
            r = epsp.shape[-1] - 1  # -1 gives the last item's index
            if last_unprocessed >= r:
                break
        depsp = torch.abs(epsp[:, last_unprocessed: r+1] - epsp_ref)  # [B, R]
        da = a[:, :-2].unsqueeze(-1) * directors[last_unprocessed: r+1].view(1, 1, -1)
        alp[:, :-1, last_unprocessed: r+1] = da - (da - alp_ref[:, :-1].unsqueeze(-1)) * torch.exp(-c[:, :-1].unsqueeze(-1) * depsp.unsqueeze(1))

        alp[:, -1, last_unprocessed: r+1] = alp_ref[:, -1].unsqueeze(-1)

        if_mask = alp_ref[:, -1].unsqueeze(-1) * directors[last_unprocessed: r+1].unsqueeze(0) + a[:, -1].unsqueeze(-1) < 0  # [B, R]
        # depsp_thr = torch.zeros_like(if_mask)  # [B, R]
        in_log = a[:, -1].unsqueeze(-1) / ((a[:, -2] - a[:, -1]).unsqueeze(-1) - directors[last_unprocessed: r + 1].unsqueeze(0) * alp[:, -1, last_unprocessed: r + 1])
        in_log[~if_mask] = 1
        depsp_thr = -1. / c[:, -1].unsqueeze(-1) * torch.log(in_log)  # todo: faster
        depsp_thr[~if_mask] = 0
        da_ = directors[last_unprocessed: r + 1] * (a[:, -2] - a[:, -1]).unsqueeze(-1)
        torch_min, _ = torch.min(torch.stack([depsp, depsp_thr]), dim=0)  # [B, R]
        alp[:, -1, last_unprocessed: r+1][if_mask] = da_[if_mask] - (da_[if_mask] - alp[:, -1, last_unprocessed: r+1][if_mask]) * torch.exp((-c[:, -1].unsqueeze(-1) * torch_min)[if_mask])
        depsp2 = 0 + depsp
        depsp2[if_mask] = depsp[if_mask] - depsp_thr[if_mask]
        depsp2[torch.logical_and(if_mask, depsp2 < 0)] = 0

        if_mask = torch.logical_and(depsp2 > 0, torch.logical_or(depsp_thr > 0, directors[last_unprocessed: r+1].unsqueeze(0) * alp[:, -1, last_unprocessed: r+1] - a[:, -1].unsqueeze(-1) <= 0))
        depsp_thr[if_mask] = ((a[:, -1].unsqueeze(-1) - directors[last_unprocessed: r+1].unsqueeze(0) * alp[:, -1, last_unprocessed: r+1]) / c[:, -1].unsqueeze(-1) / a[:, -2].unsqueeze(-1))[if_mask]
        torch_min, _ = torch.min(torch.stack([depsp2, depsp_thr]), dim=0)
        temp_alp = alp[:, -1, last_unprocessed: r + 1] + directors[last_unprocessed: r + 1].unsqueeze(0) * c[:, -1].unsqueeze(-1) * a[:, -2].unsqueeze(-1) * torch_min
        alp[:, -1, last_unprocessed: r+1][if_mask] = temp_alp[if_mask]
        depsp2[if_mask] = depsp2[if_mask] - depsp_thr[if_mask]
        depsp2[torch.logical_and(if_mask, depsp2 < 0)] = 0

        if_mask = depsp2 > 0
        daa = directors[last_unprocessed: r + 1].unsqueeze(0) * ((a[:, -2] + a[:, -1]).unsqueeze(-1))
        alp[:, -1, last_unprocessed: r+1][if_mask] = (daa - (daa - alp[:, -1, last_unprocessed: r + 1]) * torch.exp(-c[:, -1].unsqueeze(-1) * depsp2))[if_mask]

        alp_ref = alp[..., r]
        epsp_ref = epsp[:, r: r + 1]
        last_unprocessed = r + 1

    np.save('alp_t.npy', alp.detach().numpy())
    sig = SQR32 * torch.sum(alp, dim=1) + kiso
    return sig
