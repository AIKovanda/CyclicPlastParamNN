from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from taskchain import InMemoryData, Parameter, Task, DirData
from torch.utils.data import Dataset
from tqdm import trange

from rcpl.config import REAL_EXPERIMENT
from rcpl.dataset import RCLPDataset
from rcpl.rcpl import Experiment, get_random_pseudo_experiment, RandomCyclicPlasticLoadingParams, rcpl_torch_one


class DatasetInfoTask(Task):
    class Meta:
        input_tasks = []
        parameters = [
            Parameter("epsp", default=None),
            Parameter("random_experiment_params", default=None),
            Parameter("dim"),
            Parameter("kappa_dim"),
            Parameter("split"),
            Parameter("n_samples_for_stats"),
        ]

    def run(self, epsp, random_experiment_params, dim, kappa_dim, split, n_samples_for_stats) -> dict:
        assert (epsp is not None) ^ (random_experiment_params is not None)
        if epsp is not None:
            experiment = Experiment(epsp=epsp)
        else:
            experiment = Experiment.random_experiment(**random_experiment_params)

        x = np.zeros((n_samples_for_stats, len(experiment)), dtype=np.float32)
        y = np.zeros((n_samples_for_stats, 1+kappa_dim+2*dim), dtype=np.float64)
        for i in trange(n_samples_for_stats, desc='Generating random parameters'):
            if epsp is None:
                experiment = Experiment.random_experiment(**random_experiment_params)

            sig, model_params = get_random_pseudo_experiment(dim=dim, kappa_dim=kappa_dim, experiment=experiment)
            x[i] = sig
            y[i] = model_params.vector_params

        # calculate mean and std for each parameter
        param_mean = np.mean(y, axis=0)
        param_std = np.std(y, axis=0)

        x_mean = np.mean(x)
        x_std = np.std(x)

        # calculate min and max for each parameter
        param_min = np.min(y, axis=0)
        param_max = np.max(y, axis=0)

        x_min = np.min(x)
        x_max = np.max(x)

        model_params = RandomCyclicPlasticLoadingParams.generate_params(
            experiment=experiment,
            dim=dim,
            kappa_dim=kappa_dim,
        )

        return {
            'dim': dim,
            'kappa_dim': kappa_dim,
            'split': split,
            'n_samples_for_stats': n_samples_for_stats,
            'param_labels': model_params.param_labels,
            'latex_param_labels': model_params.latex_param_labels,
            'param_mean': param_mean.tolist(),
            'param_std': param_std.tolist(),
            'param_min': param_min.tolist(),
            'param_max': param_max.tolist(),
            'x_mean': x_mean.tolist(),
            'x_std': x_std.tolist(),
            'x_min': x_min.tolist(),
            'x_max': x_max.tolist(),
        }


class DatasetDirTask(Task):
    class Meta:
        input_tasks = []
        parameters = [
            Parameter("epsp", default=None),
            Parameter("random_experiment_params", default=None),
            Parameter("dim"),
            Parameter("kappa_dim"),
            Parameter("split"),
        ]

    def run(self, random_experiment_params: dict | None, epsp: list | None, dim: int, kappa_dim: int, split: tuple) -> DirData:
        assert (epsp is not None) ^ (random_experiment_params is not None)
        if epsp is not None:
            exp = Experiment(epsp=epsp)
        else:
            exp = Experiment.random_experiment(**random_experiment_params)
        data: DirData = self.get_data_object()
        dataset_dir = data.dir

        x = np.zeros((split[-1], (2 if epsp is None else 1), len(exp)), dtype=np.float32)
        y = np.zeros((split[-1], 1+kappa_dim+2*dim), dtype=np.float64)

        for i in trange(split[-1]):
            if epsp is None:
                exp = Experiment.random_experiment(**random_experiment_params)
            sig, model_params = get_random_pseudo_experiment(dim=dim, kappa_dim=kappa_dim, experiment=exp)
            x[i][0] = sig
            if epsp is None:
                assert len(exp.epsp) == len(sig)
                x[i][1] = exp.epsp
            y[i] = model_params.vector_params

        for i, name in enumerate(['train', 'val', 'test']):
            np.save(str(dataset_dir / f'{name}_x.npy'), x[split[i]:split[i + 1], :])
            np.save(str(dataset_dir / f'{name}_y.npy'), y[split[i]:split[i + 1], :])

        return data


class ExperimentTask(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = []
        parameters = [
            Parameter("epsp", default=None),
        ]

    def run(self, epsp: list | None) -> Experiment:
        exp = Experiment(epsp=epsp)
        return exp


class UnscaleParamsTask(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = [
            DatasetInfoTask,
        ]
        parameters = [
            Parameter("scale_type", default=None),
        ]

    def run(self, dataset_info: dict, scale_type: str | None) -> Callable:
        if scale_type is None:
            return lambda x: np.clip(x, a_min=1e-9, a_max=None)
        if scale_type == "standard":
            mean = np.array(dataset_info['param_mean'])
            std = np.array(dataset_info['param_std'])
            return lambda x: np.clip(x * std + mean, a_min=1e-9, a_max=None)
        elif scale_type == "minmax":
            min_ = np.array(dataset_info['param_min'])
            max_ = np.array(dataset_info['param_max'])
            return lambda x: np.clip(x * (max_ - min_) + min_, a_min=1e-9, a_max=None)
        else:
            raise ValueError(f"Unknown scale_type: {scale_type}")


class UnscaleParamsTorchTask(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = [
            DatasetInfoTask,
        ]
        parameters = [
            Parameter("scale_type", default=None),
            Parameter("device", default="cuda", ignore_persistence=True),
        ]

    def run(self, dataset_info: dict, scale_type: str | None, device: str) -> Callable:
        if scale_type is None:
            return lambda x: torch.clamp(x, min=1e-9, max=None)
        if scale_type == "standard":
            mean = torch.tensor(dataset_info['param_mean'], device=device)
            std = torch.tensor(dataset_info['param_std'], device=device)
            return lambda x: torch.clamp(x * std + mean, min=1e-9, max=None)
        elif scale_type == "minmax":
            min_ = torch.tensor(dataset_info['param_min'], device=device)
            max_ = torch.tensor(dataset_info['param_max'], device=device)
            return lambda x: torch.clamp(x * (max_ - min_) + min_, min=1e-9, max=None)
        else:
            raise ValueError(f"Unknown scale_type: {scale_type}")


class GetRandomPseudoExperimentTask(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = [
            UnscaleParamsTask,
        ]
        parameters = [
            Parameter("dim"),
            Parameter("kappa_dim"),
            Parameter("scale_type", default=None),
        ]

    def run(self, unscale_params: callable, dim: int, kappa_dim: int) -> Callable:
        def get_random_pseudo_experiment_(scaled_params, experiment=None):
            params = unscale_params(scaled_params)
            return get_random_pseudo_experiment(dim=dim, kappa_dim=kappa_dim, experiment=experiment, vector_params=params)
        return get_random_pseudo_experiment_


class AbstractDatasetTask(Task):
    class Meta:
        abstract = True

    def run(self, epsp: list | None, dataset_dir: Path, dataset_info: dict, scale_type: str) -> Dataset:
        x = torch.from_numpy(np.load(dataset_dir / f'{self.meta.split_name}_x.npy'))
        y = torch.from_numpy(np.load(dataset_dir / f'{self.meta.split_name}_y.npy'))

        if scale_type is not None:
            if scale_type == "standard":
                mean = np.array(dataset_info['param_mean'])
                std = np.array(dataset_info['param_std'])
                y -= mean
                y /= std

            elif scale_type == "minmax":
                min_ = np.array(dataset_info['param_min'])
                max_ = np.array(dataset_info['param_max'])
                y -= min_
                y /= (max_ - min_)
            else:
                raise ValueError(f"Unknown scale_type: {scale_type}")

        return RCLPDataset(x=x, y=y.float(), epsp=(np.array(epsp) if epsp is not None else None))


class TrainDatasetTask(AbstractDatasetTask):
    class Meta:
        data_class = InMemoryData
        input_tasks = [DatasetDirTask, DatasetInfoTask]
        parameters = [
            Parameter("epsp", default=None),
            Parameter("scale_type", default=None),
        ]
        split_name = 'train'


class ValDatasetTask(AbstractDatasetTask):
    class Meta:
        data_class = InMemoryData
        input_tasks = [DatasetDirTask, DatasetInfoTask]
        parameters = [
            Parameter("epsp", default=None),
            Parameter("scale_type", default=None),
        ]
        split_name = 'val'


class TestDatasetTask(AbstractDatasetTask):
    class Meta:
        data_class = InMemoryData
        input_tasks = [DatasetDirTask, DatasetInfoTask]
        parameters = [
            Parameter("epsp", default=None),
            Parameter("scale_type", default=None),
        ]
        split_name = 'test'


class GetCrlbTask(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = [
            UnscaleParamsTask,
        ]
        parameters = [
            Parameter("dim"),
            Parameter("kappa_dim"),
            Parameter("scale_type", default=None),
        ]

    def run(self, unscale_params: callable, dim: int, kappa_dim: int) -> Callable:

        def get_crlb_(epsp, unscaled_params):
            assert epsp is not None
            assert len(unscaled_params.shape) == 1
            theta = unscaled_params.detach().cpu().clone().type(torch.float64)
            theta.requires_grad = True
            sig_pred = rcpl_torch_one(epsp=torch.tensor(epsp, dtype=torch.float64), theta=theta, dim=dim, kappa_dim=kappa_dim)

            grad_a = torch.zeros((len(theta), len(sig_pred)), dtype=torch.float64)
            for id_ in trange(len(sig_pred), desc='Calculating gradients of theta'):
                if theta.grad is not None:
                    theta.grad.zero_()

                # Backpropagate for the current element
                sig_pred[id_].backward(retain_graph=True)
                grad_a[:, id_] = theta.grad

            fm_a = grad_a @ grad_a.T
            # regularize
            fm_a += torch.eye(len(theta)) * 1e-9
            var = torch.inverse(fm_a).diag()
            return {
                'theta': theta.detach().cpu().numpy(),
                'var': var.detach().cpu().numpy(),
                'std': np.sqrt(var.detach().cpu().numpy()),
                'relative_std': np.sqrt(var.detach().cpu().numpy()) / theta.detach().cpu().numpy(),
            }
        return get_crlb_
