from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
from rcpl.dataset import RCLPDataset
from rcpl.rcpl import Experiment, get_random_pseudo_experiment
from taskchain import InMemoryData, Parameter, Task, DirData
from torch.utils.data import Dataset
from tqdm import trange


class DatasetDir(Task):
    class Meta:
        input_tasks = []
        parameters = [
            Parameter("epsp", default=None),
            Parameter("dim"),
            Parameter("kappa_dim"),
            Parameter("split"),
        ]

    def run(self, epsp: list | None, dim: int, kappa_dim: int, split: tuple) -> DirData:
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
            y[i] = model_params.scaled_params

        for i, name in enumerate(['train', 'val', 'test']):
            np.save(str(dataset_dir / f'{name}_x.npy'), x[split[i]:split[i + 1], :])
            np.save(str(dataset_dir / f'{name}_y.npy'), y[split[i]:split[i + 1], :])

        return data


class GetRandomPseudoExperiment(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = []
        parameters = [
            Parameter("dim"),
            Parameter("kappa_dim"),
        ]

    def run(self, dim: int, kappa_dim: int) -> Callable:
        return partial(get_random_pseudo_experiment, dim=dim, kappa_dim=kappa_dim)


class TrainDataset(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = [DatasetDir]
        parameters = [
            Parameter("epsp", default=None),
        ]

    def run(self, epsp: list | None, dataset_dir: Path) -> Dataset:
        exp = Experiment(epsp=epsp)
        return RCLPDataset(exp, 'train', dataset_dir=dataset_dir)


class ValDataset(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = [DatasetDir]
        parameters = [
            Parameter("epsp", default=None),
        ]

    def run(self, epsp: list | None, dataset_dir: Path) -> Dataset:
        exp = Experiment(epsp=epsp)
        return RCLPDataset(exp, 'val', dataset_dir=dataset_dir)


class TestGen(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = [DatasetDir]
        parameters = [
            Parameter("epsp", default=None),
        ]

    def run(self, epsp: list | None, dataset_dir: Path) -> Dataset:
        exp = Experiment(epsp=epsp)
        return RCLPDataset(exp, 'test', dataset_dir=dataset_dir)
