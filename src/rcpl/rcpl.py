# import abc
# from functools import cache
#
# import numpy as np
# import torch
# from numba import jit
#
# SQR32 = np.sqrt(1.5)
#
#
# class RandomCyclicPlasticLoadingParams(abc.ABC):
#     """
#     This class is used to store parameters of the random cyclic plastic loading model. It is used to generate
#     random parameters for cyclic plastic loading and to scale and unscale parameters (normalization).
#     """
#
#     def __init__(self, theta: dict | np.ndarray, dim: int, kappa_dim=2, experiment=None, use_torch=False,
#                  num_model='MAFTr'):
#         self.uses_torch = use_torch
#         self.experiment = experiment
#         self.dim = dim
#         self.kappa_dim = kappa_dim
#         assert num_model in ['MAF, MAFTr']
#         self.num_model = num_model
#         self.uses_theta = {
#             'k0': {'dim': 1, 'label': 'k0', 'latex_label': 'k_0'},
#             'kap': {'dim': kappa_dim, 'label': 'κ', 'latex_label': '\\kappa'},
#             'c': {'dim': dim, 'label': 'c', 'latex_label': 'c'},
#             'a': {'dim': dim, 'label': 'a', 'latex_label': 'a'},
#         }
#         if num_model == 'MAFTr':
#             self.uses_theta['aL'] = {'dim': 1, 'label': 'aL', 'latex_label': '\\overline{a}'}
#         self.theta = theta if isinstance(theta, dict) else self.vector_theta_to_dict(theta)
#
#     @property
#     def labels(self, latex=False):
#         label_key = 'latex_label' if latex else 'label'
#         labels = []
#         for param_name, param_info in self.uses_theta.items():
#             if param_info['dim'] == 1:
#                 labels.append(param_info[label_key])
#             else:
#                 for i in range(param_info['dim']):
#                     labels.append(f'{param_info[label_key]}{i + 1}')
#         return labels
#
#     @property
#     def vector_theta(self):
#         return np.concatenate([self.theta[i] for i in self.uses_theta.keys()])
#
#     def vector_theta_to_dict(self, theta):
#         assert len(theta) == self.theta_len
#         param_cumsum = np.cumsum([param_info['dim'] for param_info in self.uses_theta.values()])
#         return {
#             param_name: theta[param_cumsum[i]:param_cumsum[i+1]]
#             for i, param_name in enumerate(self.uses_theta.keys())
#         }
#
#     @property
#     @cache
#     def theta_len(self):
#         return sum([param_info['dim'] for param_info in self.uses_theta.values()])
#
#     @classmethod
#     def generate_theta(cls, dim: int, kappa_dim: int = 2, experiment=None, use_torch=False):
#         """
#         Generates random parameters for the cyclic plastic loading model. Returns an instance of this class.
#         """
#         if kappa_dim != 2:
#             raise NotImplementedError
#
#         a = np.random.uniform(0, 1, size=dim)
#         a /= np.sum(a)
#         a = np.random.uniform(150, 350) * a
#
#         theta = {
#             'k0': np.random.uniform(15, 250, size=1),
#             'kap': np.array([np.random.uniform(100, 10000), 1 / np.random.uniform(30, 150)]),
#             'c': np.sort([np.exp(np.random.uniform(np.log(1000), np.log(10000))),
#                           *np.exp(np.random.uniform(np.log(50), np.log(2000), size=dim - 1))])[::-1],
#             'a': a,
#         }
#         if use_torch:
#             theta = {key: torch.from_numpy(value.copy()) for key, value in theta.items()}
#
#         return cls(
#             theta=theta,
#             experiment=experiment,
#             use_torch=use_torch,
#             dim=dim,
#             kappa_dim=kappa_dim,
#         )
#
#
# def random_cyclic_plastic_loading(k0: np.ndarray, kap: np.ndarray, a: np.ndarray, c: np.ndarray, epsp: np.ndarray):
#     epspc = np.zeros_like(epsp)
#     epspc[1:] = np.cumsum(np.abs(epsp[1:] - epsp[:-1]))
#
#     directors = np.append(0, np.sign(epsp[1:] - epsp[:-1]))
#     signs = np.sign(directors)
#     # Find where the product of consecutive signs is -1 (indicating a sign change)
#     reversal_ids = np.append(np.where(signs[:-1] * signs[1:] == -1)[0], -1)  # -1 so that index never reaches it
#     return random_cyclic_plastic_loading_engine(k0, kap, a, c, epsp, epspc, directors, reversal_ids)
#
#
# def random_cyclic_plastic_loading_el(k0: np.ndarray, kap: np.ndarray, a: np.ndarray, c: np.ndarray, epsp: np.ndarray):
#     epspc = np.zeros_like(epsp)
#     epspc[1:] = np.cumsum(np.abs(epsp[1:] - epsp[:-1]))
#
#     directors = np.append(0, np.sign(epsp[1:] - epsp[:-1]))
#     signs = np.sign(directors)
#     # Find where the product of consecutive signs is -1 (indicating a sign change)
#     reversal_ids = np.append(np.where(signs[:-1] * signs[1:] == -1)[0], -1)  # -1 so that index never reaches it
#     return random_cyclic_plastic_loading_engine_el(k0, kap, a, c, epsp, epspc, directors, reversal_ids)
#
#
# @jit(nopython=True)  # this makes the function much faster by compiling it to machine code
# def random_cyclic_plastic_loading_engine(k0: np.ndarray, kap: np.ndarray, a: np.ndarray, c: np.ndarray,
#                                          epsp: np.ndarray, epspc: np.ndarray,
#                                          directors: np.ndarray, reversal_ids: np.ndarray):
#     kiso = directors / kap[1] * (1 - (1 - k0[0] * kap[1]) * np.exp(-SQR32 * kap[0] * kap[
#         1] * epspc))  # Vyvoj kiso jako funkce epspc, rovnou orientovane ve smeru narustajici plasticke deformace.
#     #    kiso = kiso + directors/kap[4]*(1-np.exp(-SQR32 * kap[3] * kap[4] * epspc))  # Odkomentuj k aktivaci druhe funkce isotropniho zpevneni
#     #    kiso = kiso + directors/kap[6]*(1-np.exp(-SQR32 * kap[5] * kap[6] * epspc))  # Odkomentuj k aktivaci treti funkce isotropniho zpevneni
#     assert len(kiso) == len(epsp)
#
#     alp = np.zeros((len(a), len(epsp)), dtype=np.float64)
#     alp_ref = np.zeros(len(a), dtype=np.float64)
#     epsp_ref = epsp[0]
#     k = 0
#     for i, epsp_i in enumerate(epsp):
#         # Spocita absolutni hodnotu noveho prirustku plasticke deformace v aktualnim segmentu
#         depsp = np.abs(epsp_i - epsp_ref)
#         # Vyvoj backstressu od posledni referencni hodnoty.
#         alp[:, i] = directors[i] * a - (directors[i] * a - alp_ref) * np.exp(-c * depsp)
#         # Aktualizuje referencni hodnoty backstressu na zacatku noveho segmentu
#         if i == reversal_ids[k]:
#             alp_ref = alp[:, i]
#             epsp_ref = epsp_i
#             k += 1
#
#     sig = SQR32 * np.sum(alp, axis=0) + kiso  # Celkova napetova odezva modelu.
#     return sig
#
#
# def random_cyclic_plastic_loading_engine_el(k0: np.ndarray, kap: np.ndarray, a: np.ndarray, c: np.ndarray,
#                                             epsp: np.ndarray, epspc: np.ndarray,
#                                             directors: np.ndarray, reversal_ids: np.ndarray):
#     kiso = directors / kap[1] * (1 - (1 - k0[0] * kap[1]) * np.exp(-SQR32 * kap[0] * kap[
#         1] * epspc))  # Vyvoj kiso jako funkce epspc, rovnou orientovane ve smeru narustajici plasticke deformace.
#     #    kiso = kiso + directors/kap[4]*(1-np.exp(-SQR32 * kap[3] * kap[4] * epspc))  # Odkomentuj k aktivaci druhe funkce isotropniho zpevneni
#     #    kiso = kiso + directors/kap[6]*(1-np.exp(-SQR32 * kap[5] * kap[6] * epspc))  # Odkomentuj k aktivaci treti funkce isotropniho zpevneni
#     assert len(kiso) == len(epsp)
#
#     alp = np.zeros((len(a)-1, len(epsp)), dtype=np.float64)
#     alp_ref = np.zeros(len(a), dtype=np.float64)
#     epsp_ref = epsp[0]
#     k = 0
#     depsp_thr = 0
#     for i, epsp_i in enumerate(epsp):
#         # Spocita absolutni hodnotu noveho prirustku plasticke deformace v aktualnim segmentu
#         depsp = np.abs(epsp_i - epsp_ref)
#         # Vyvoj backstressu od posledni referencni hodnoty.
#         alp[0:3, i] = directors[i] * a[0:3] - (directors[i] * a[0:3] - alp_ref[0:3]) * np.exp(-c[0:3] * depsp)
#         # Prepisu posledni segment backstressu
#         alp[3, i] = alp_ref[3]
#         if directors[i] * alp[3, i] < -a[4]:  # abar
#             depsp_thr = -1. / c[3] * np.log(a[3] / (a[3] - a[4] - directors[i] * alp[3, i]))
#             alp[:, 3] = directors[i] * (a[3] - a[4]) - (directors * (a[3] - a[4]) - alp[3]) * np.exp(-c[3] * np.min(depsp, depsp_thr))
#             depsp = np.max(0, depsp - depsp_thr)
#         if depsp > 0 and (depsp_thr > 0 or directors[i] * alp[3, i] <= a[4]):
#             depsp_thr = (a[4] - directors[i] * alp[3, i]) / c[3] / a[3]
#             alp[3, i] = alp[3, i] + directors[i] * c[3] * a[3] * min(depsp, depsp_thr)
#             depsp = max(0, depsp - depsp_thr)
#         if depsp > 0:
#             alp[3] = directors[i] * (a[3] + a[4]) - (directors[i] * (a[3] + a[4]) - alp[3]) * np.exp(-c[3] * depsp)
#         # Aktualizuje referencni hodnoty backstressu na zacatku noveho segmentu
#         if i == reversal_ids[k]:
#             alp_ref = alp[:, i]
#             epsp_ref = epsp_i
#             k += 1
#
#     sig = SQR32 * np.sum(alp, axis=0) + kiso  # Celkova napetova odezva modelu.
#     return sig, alp, kiso
#
#
# def random_cyclic_plastic_loading_torch_batch(k0: torch.Tensor, kap: torch.Tensor, a: torch.Tensor, c: torch.Tensor,
#                                               epsp: torch.Tensor):
#     """
#     Each input parameter is a tensor of shape (batch_size, parameter_size), including EPSP. Epsp is expected to have
#     the same number of points in each segment so that directors, reversal_ids are identical for all samples in the batch.
#     """
#     epspc = torch.zeros_like(epsp, device=epsp.device)
#     epspc[:, 1:] = torch.cumsum(torch.abs(epsp[:, 1:] - epsp[:, :-1]), dim=1)
#
#     directors = torch.cat((torch.tensor([0], device=epsp.device), torch.sign(
#         epsp[0, 1:] - epsp[0, :-1])))  # taking the first sample in the batch - the rest must be the same
#     signs = torch.sign(directors)
#     # Find where the product of consecutive signs is -1 (indicating a sign change)
#     reversal_ids = torch.cat(
#         (torch.where(signs[:-1] * signs[1:] == -1)[0],
#          torch.tensor([-1], device=epsp.device)))  # -1 so that index never reaches it
#     return random_cyclic_plastic_loading_engine_t(k0, kap, a, c, epsp, epspc, directors, reversal_ids)
#
#
# def random_cyclic_plastic_loading_engine_t(k0: torch.Tensor, kap: torch.Tensor, a: torch.Tensor, c: torch.Tensor,
#                                            epsp: torch.Tensor, epspc: torch.Tensor,
#                                            directors: torch.Tensor, reversal_ids: torch.Tensor):
#     kiso = directors.repeat(kap.shape[0], 1) / kap[:, 1:2] * (
#                 1 - (1 - k0 * kap[:, 1:2]) * torch.exp(-SQR32 * kap[:, 0:1] * kap[:, 1:2] * epspc))
#     alp = torch.zeros((*a.shape, epsp.shape[-1]), dtype=kap.dtype, device=epsp.device)
#     alp_ref = torch.zeros(*a.shape, dtype=kap.dtype, device=epsp.device)
#     epsp_ref = epsp[:, :1]
#     k = 0
#     for i in range(epsp.shape[-1]):
#         depsp = torch.abs(epsp[:, i: i + 1] - epsp_ref)
#         alp[..., i] = directors[i] * a - (directors[i] * a - alp_ref) * torch.exp(-c * depsp)
#         if i == reversal_ids[k]:  # Use .item() to extract the value from the tensor
#             alp_ref = alp[..., i]
#             epsp_ref = epsp[:, i: i + 1]
#             k += 1
#
#     sig = SQR32 * torch.sum(alp, dim=1) + kiso
#     return sig
#
#
# @torch.compile
# def predict_stress_torch(theta: torch.Tensor, epsp: torch.Tensor, dim, kappa_dim) -> torch.Tensor:
#     return random_cyclic_plastic_loading_torch_batch(
#         epsp=torch.unsqueeze(epsp, 0),
#         k0=torch.unsqueeze(theta[:1], 0),
#         kap=torch.unsqueeze(theta[1:1 + kappa_dim], 0),
#         c=torch.unsqueeze(theta[1 + kappa_dim:1 + kappa_dim + dim], 0),
#         a=torch.unsqueeze(theta[1 + kappa_dim + dim:], 0),
#     )[0]
#
#
# # @torch.compile
# def predict_stress_torch_batch(theta: torch.Tensor, epsp: torch.Tensor, dim, kappa_dim) -> torch.Tensor:
#     return random_cyclic_plastic_loading_torch_batch(
#         epsp=epsp,
#         k0=theta[:, :1],
#         kap=theta[:, 1:1 + kappa_dim],
#         c=theta[:, 1 + kappa_dim:1 + kappa_dim + dim],
#         a=theta[:, 1 + kappa_dim + dim:],
#     )
#
#
# def get_random_pseudo_experiment(dim, kappa_dim, experiment: Experiment, vector_theta: np.ndarray = None):
#     if vector_theta is None:
#         model_theta = RandomCyclicPlasticLoadingParams.generate_theta(
#             experiment=experiment,
#             dim=dim,
#             kappa_dim=kappa_dim,
#         )
#     else:
#         model_theta = RandomCyclicPlasticLoadingParams(
#             theta=vector_theta,
#             experiment=experiment,
#             dim=dim,
#             kappa_dim=kappa_dim,
#         )
#     sig = random_cyclic_plastic_loading(**model_theta.theta, epsp=experiment.epsp)
#     # if sig.max() > 5000:
#     #     print(f'Parameters: {model_theta.params}. Max of sig is {sig.max()}.')
#     return sig, model_params
