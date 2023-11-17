from pathlib import Path
from typing import Callable

import numpy as np
import torch
from taskchain import InMemoryData, Parameter, ModuleTask, DirData
from torch.utils.data import Dataset
from tqdm import trange

from rcpl.dataset import CPLDataset
from rcpl.experiment import Experiment, RandomExperimentGenerator
from rcpl.cpl_models import CPLModelFactory


class DatasetInfoTask(ModuleTask):
    class Meta:
        input_tasks = []
        parameters = [
            Parameter("experiment", default=None),
            Parameter("exp_generator", default=None),
            Parameter("exp_representation"),
            Parameter("model_factory"),
            Parameter("split"),
            Parameter("n_samples_for_stats"),
        ]

    def run(self, experiment, exp_generator: RandomExperimentGenerator, exp_representation,
            model_factory: CPLModelFactory, split, n_samples_for_stats) -> dict:

        if experiment is None:
            experiment = exp_generator.generate_representation(exp_representation)

        x = np.zeros((n_samples_for_stats, len(experiment.get_epsp_representation(exp_representation))), dtype=np.float32)
        random_num_model = model_factory.make_model()
        y = np.zeros((n_samples_for_stats, random_num_model.theta_len), dtype=np.float64)
        for i in trange(n_samples_for_stats, desc='Generating random parameters'):
            if experiment is None:
                experiment = exp_generator.generate_representation(exp_representation)

            num_model = model_factory.make_model()
            x[i] = num_model.predict_stress(experiment.get_epsp_representation(exp_representation))
            y[i] = num_model.theta

        return {
            'apriori_params_bound': model_factory.params_bound,
            'apriori_params_lower_bound': model_factory.lower_bound.tolist(),
            'apriori_params_upper_bound': model_factory.upper_bound.tolist(),
            'model_kwargs': model_factory.model_kwargs,
            'split': split,
            'n_samples_for_stats': n_samples_for_stats,
            'labels': random_num_model.labels(),
            'latex_labels': random_num_model.labels(latex=True),
            'param_mean': np.mean(y, axis=0).tolist(),
            'param_std': np.std(y, axis=0).tolist(),
            'param_min': np.min(y, axis=0).tolist(),
            'param_max': np.max(y, axis=0).tolist(),
            'x_mean': np.mean(x).tolist(),
            'x_std': np.std(x).tolist(),
            'x_min': np.min(x).tolist(),
            'x_max': np.max(x).tolist(),
            'exp_representation': list(exp_representation),
        }


class ModelFactoryTask(ModuleTask):
    class Meta:
        data_class = InMemoryData
        input_tasks = []
        parameters = [
            Parameter("model_factory"),
        ]

    def run(self, model_factory: CPLModelFactory) -> CPLModelFactory:
        return model_factory


class DatasetDirTask(ModuleTask):
    class Meta:
        input_tasks = []
        parameters = [
            Parameter("experiment", default=None),
            Parameter("exp_generator", default=None),
            Parameter("exp_representation"),
            Parameter("model_factory"),
            Parameter("split"),
        ]

    def run(self, experiment, exp_generator: RandomExperimentGenerator, exp_representation,
            model_factory: CPLModelFactory, split: tuple) -> DirData:
        assert (experiment is not None) ^ (exp_generator is not None)
        data: DirData = self.get_data_object()
        dataset_dir = data.dir

        generate_epsp = experiment is None
        if experiment is None:
            experiment = exp_generator.generate_representation(exp_representation)

        x = np.zeros((split[-1], (2 if generate_epsp else 1), len(experiment.get_epsp_representation(exp_representation))), dtype=np.float32)
        random_num_model = model_factory.make_model()
        y = np.zeros((split[-1], random_num_model.theta_len), dtype=np.float64)

        for i in trange(split[-1]):
            if experiment is None:
                experiment = exp_generator.generate_representation(exp_representation)

            num_model = model_factory.make_model()
            epsp = experiment.get_epsp_representation(exp_representation)
            stress = num_model.predict_stress(epsp)
            assert len(epsp) == len(stress)
            x[i, 0] = stress
            if generate_epsp:
                x[i, 1] = epsp
            y[i] = num_model.theta

        for i, name in enumerate(['train', 'val', 'test']):
            np.save(str(dataset_dir / f'{name}_x.npy'), x[split[i]:split[i + 1], :])
            np.save(str(dataset_dir / f'{name}_y.npy'), y[split[i]:split[i + 1], :])

        return data


class UnscaleThetaTask(ModuleTask):
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


class UnscaleThetaTorchTask(ModuleTask):
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


class ScaleThetaTorchTask(ModuleTask):
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
            return lambda x: (x - mean) / std
        elif scale_type == "minmax":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown scale_type: {scale_type}")


class GetRandomPseudoExperimentTask(ModuleTask):
    class Meta:
        data_class = InMemoryData
        input_tasks = [
            UnscaleThetaTask,
        ]
        parameters = [
            Parameter("model_factory"),
        ]

    def run(self, unscale_theta: callable, model_factory: CPLModelFactory) -> Callable:
        def get_random_pseudo_experiment_(scaled_theta, epsp_correct_representation: np.ndarray):
            theta = unscale_theta(scaled_theta)
            num_model = model_factory.make_model(theta)
            return num_model.predict_stress(epsp_correct_representation)
        return get_random_pseudo_experiment_


class AbstractDatasetTask(ModuleTask):
    class Meta:
        abstract = True

    def run(self, experiment: Experiment, exp_representation, dataset_dir: Path, dataset_info: dict,
            scale_type: str) -> Dataset:
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

        return CPLDataset(x=x, y=y.float(), epsp=(experiment.get_epsp_representation(exp_representation)
                                                  if experiment is not None else None))


class TrainDatasetTask(AbstractDatasetTask):
    class Meta:
        data_class = InMemoryData
        input_tasks = [DatasetDirTask, DatasetInfoTask]
        parameters = [
            Parameter("experiment", default=None),
            Parameter("exp_representation"),
            Parameter("scale_type", default=None),
        ]
        split_name = 'train'


class ValDatasetTask(AbstractDatasetTask):
    class Meta:
        data_class = InMemoryData
        input_tasks = [DatasetDirTask, DatasetInfoTask]
        parameters = [
            Parameter("experiment", default=None),
            Parameter("exp_representation"),
            Parameter("scale_type", default=None),
        ]
        split_name = 'val'


class TestDatasetTask(AbstractDatasetTask):
    class Meta:
        data_class = InMemoryData
        input_tasks = [DatasetDirTask, DatasetInfoTask]
        parameters = [
            Parameter("experiment", default=None),
            Parameter("exp_representation"),
            Parameter("scale_type", default=None),
        ]
        split_name = 'test'


class GetCrlbTask(ModuleTask):
    class Meta:
        data_class = InMemoryData
        input_tasks = []
        parameters = [
            Parameter("model_factory"),
            Parameter("scale_type", default=None),
        ]

    def run(self, model_factory: CPLModelFactory, scale_type: str | None) -> Callable:

        def get_crlb_(epsp, unscaled_theta):
            assert epsp is not None
            assert len(unscaled_theta.shape) == 1
            unscaled_theta = unscaled_theta.detach().cpu().clone().type(torch.float64)
            unscaled_theta.requires_grad = True
            num_model = model_factory.make_model(unscaled_theta)
            sig_pred = num_model.predict_stress_torch(epsp=torch.tensor(epsp, dtype=torch.float64))

            grad_a = torch.zeros((len(unscaled_theta), len(sig_pred)), dtype=torch.float64)
            for id_ in trange(len(sig_pred), desc='Calculating gradients of theta'):
                if unscaled_theta.grad is not None:
                    unscaled_theta.grad.zero_()

                # Backpropagate for the current element
                sig_pred[id_].backward(retain_graph=True)
                grad_a[:, id_] = unscaled_theta.grad

            fm_a = grad_a @ grad_a.T
            # regularize
            fm_a += torch.eye(len(unscaled_theta)) * 1e-9
            var = torch.inverse(fm_a).diag()
            return {
                'theta': unscaled_theta.detach().cpu().numpy(),
                'var': var.detach().cpu().numpy(),
                'std': np.sqrt(var.detach().cpu().numpy()),
                'relative_std': np.sqrt(var.detach().cpu().numpy()) / unscaled_theta.detach().cpu().numpy(),
            }
        return get_crlb_
