from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
from rcpl.dataset import RCLPDataset
from rcpl.rcpl import Experiment, get_random_pseudo_experiment, RandomCyclicPlasticLoadingParams
from taskchain import InMemoryData, Parameter, Task, DirData
from torch.utils.data import Dataset
from tqdm import trange


class DatasetInfoTask(Task):
    class Meta:
        input_tasks = []
        parameters = [
            Parameter("epsp", default=None),
            Parameter("dim"),
            Parameter("kappa_dim"),
            Parameter("split"),
            Parameter("n_samples_for_stats"),
            Parameter("scale_type", default=None),
        ]

    def run(self, epsp, dim, kappa_dim, split, n_samples_for_stats, scale_type) -> dict:
        experiment = Experiment(epsp=epsp)
        param_arr = np.zeros((n_samples_for_stats, 1+kappa_dim+2*dim), dtype=np.float64)
        for i in trange(n_samples_for_stats, desc='Generating random parameters'):
            if epsp is None:
                experiment = Experiment(epsp=epsp)
            model_params = RandomCyclicPlasticLoadingParams.generate_params(
                experiment=experiment,
                dim=dim,
                kappa_dim=kappa_dim,
            )
            param_arr[i] = model_params.vector_params

        # calculate mean and std for each parameter
        param_mean = np.mean(param_arr, axis=0)
        param_std = np.std(param_arr, axis=0)

        # calculate min and max for each parameter
        param_min = np.min(param_arr, axis=0)
        param_max = np.max(param_arr, axis=0)

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
            'scale_type': scale_type,
        }


class DatasetDirTask(Task):
    class Meta:
        input_tasks = [
            DatasetInfoTask,
        ]
        parameters = [
            Parameter("epsp", default=None),
            Parameter("dim"),
            Parameter("kappa_dim"),
            Parameter("split"),
            Parameter("scale_type", default=None),
        ]

    def run(self, dataset_info: dict, epsp: list | None, dim: int, kappa_dim: int, split: tuple, scale_type: str | None) -> DirData:
        exp = Experiment(epsp=epsp)
        data: DirData = self.get_data_object()
        dataset_dir = data.dir

        x = np.zeros((split[-1], len(exp)), dtype=np.float32)
        y = np.zeros((split[-1], 1+kappa_dim+2*dim), dtype=np.float32)

        for i in trange(split[-1]):
            if epsp is None:
                exp = Experiment(epsp=epsp)
            sig, model_params = get_random_pseudo_experiment(dim=dim, kappa_dim=kappa_dim, experiment=exp)
            x[i] = sig
            y[i] = model_params.vector_params

        if scale_type is not None:
            if scale_type == "standard":
                mean = np.array(dataset_info['param_mean'])
                std = np.array(dataset_info['param_std'])
                y = (y - mean) / std

            elif scale_type == "minmax":
                min_ = np.array(dataset_info['param_min'])
                max_ = np.array(dataset_info['param_max'])
                y = (y - min_) / (max_ - min_)
            else:
                raise ValueError(f"Unknown scale_type: {scale_type}")

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


class TrainDatasetTask(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = [DatasetDirTask]
        parameters = [
            Parameter("epsp", default=None),
        ]

    def run(self, epsp: list | None, dataset_dir: Path) -> Dataset:
        exp = Experiment(epsp=epsp)
        return RCLPDataset(exp, 'train', dataset_dir=dataset_dir)


class ValDatasetTask(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = [DatasetDirTask]
        parameters = [
            Parameter("epsp", default=None),
        ]

    def run(self, epsp: list | None, dataset_dir: Path) -> Dataset:
        exp = Experiment(epsp=epsp)
        return RCLPDataset(exp, 'val', dataset_dir=dataset_dir)


class TestGen(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = [DatasetDirTask]
        parameters = [
            Parameter("epsp", default=None),
        ]

    def run(self, epsp: list | None, dataset_dir: Path) -> Dataset:
        exp = Experiment(epsp=epsp)
        return RCLPDataset(exp, 'test', dataset_dir=dataset_dir)
