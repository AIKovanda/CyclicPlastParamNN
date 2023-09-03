import json
from itertools import chain
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from taskchain import Parameter, Task, DirData, InMemoryData
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rcpl.config import RUNS_DIR
from rcpl.rcpl import Experiment
from rcpl.reporter import Reporter
from rcpl.tasks.dataset import TrainDatasetTask, ValDatasetTask, GetRandomPseudoExperimentTask, DatasetInfoTask

from functools import partial
from rcpl import metrics, models


def get_metrics(metrics_config: dict | None):
    if metrics_config is None:
        return {}
    return {key: eval(value) for key, value in metrics_config.items()}


class TrainModelTask(Task):
    class Meta:
        input_tasks = [
            DatasetInfoTask,
            TrainDatasetTask,
            ValDatasetTask,
            GetRandomPseudoExperimentTask,
        ]
        parameters = [
            Parameter("architecture"),
            Parameter("model_parameters"),
            Parameter("optim", default='AdamW'),
            Parameter("optim_kwargs"),
            Parameter("epochs"),
            Parameter("scheduler", default='None'),
            Parameter("exec_str", default=None),
            Parameter("save_metrics_n", default=100, ignore_persistence=True),
            Parameter("evaluate_n", default=300, ignore_persistence=True),
            Parameter("checkpoint_n", default=2000),
            Parameter("batch_size"),
            Parameter("valid_batch_size", ignore_persistence=True),
            Parameter("shuffle", default=True),
            Parameter("num_workers", default=-1, ignore_persistence=True),
            Parameter("drop_last", default=True),
            Parameter("do_compile", default=True, ignore_persistence=True),
            Parameter("run_dir", ignore_persistence=True),
            Parameter("run_name", ignore_persistence=True),
            Parameter("device", default="cuda", ignore_persistence=True),
            Parameter("x_metrics", default=None, ignore_persistence=True),
            Parameter("y_metrics", default=None, ignore_persistence=True),
        ]

    def try_decorator(self, func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except:
                print(f'Error in {func.__name__}. Exiting...')
                return self.get_data_object()

        return wrapper

    @staticmethod
    def evaluate_model(valid_batch_size, dataset_info, x_metrics, y_metrics, net, val_dataset, device,
                       num_workers, drop_last, get_random_pseudo_experiment):
        net.eval()

        param_labels = dataset_info['param_labels']
        train_ldr = DataLoader(val_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=num_workers,
                               drop_last=drop_last, pin_memory=True)

        reporter = Reporter(batch_size=valid_batch_size)

        loss_func = nn.MSELoss()
        for batch_id, (batch_x, batch_y, epsp) in enumerate(train_ldr):
            x_true = batch_x.to(device)
            y_true = batch_y.to(device)
            with torch.no_grad():
                y_pred = net(x_true)
                reporter.add_deque('Loss/val/_all', loss_func(y_pred, y_true).item(), use_max_len=False)
                for metric_name, metric_func in y_metrics.items():
                    reporter.add_deque(f'{metric_name}/val', metric_func(y_pred=y_pred, y_true=y_true), use_max_len=False)

                if param_labels is not None:
                    for param_value, param_name in zip(torch.mean((y_pred - y_true) ** 2, dim=0).cpu().numpy().tolist(), param_labels):
                        reporter.add_deque(f'{param_name}/val', param_value, use_max_len=False)

                # comparing x_pred and x_true
                for sample_id, (x_i_true, y_i_pred_npy, epsp_i) in enumerate(zip(x_true.detach().cpu(), y_pred.detach().cpu().numpy(), epsp.cpu().numpy())):
                    x_i_pred = torch.unsqueeze(torch.from_numpy(get_random_pseudo_experiment(scaled_params=y_i_pred_npy, experiment=Experiment(epsp_i))[0]), 0)
                    for metric_name, metric_func in x_metrics.items():
                        reporter.add_deque(f'{metric_name}/val', metric_func(x_pred=x_i_pred, x_true=x_i_true), use_max_len=False)
        net.train()
        return reporter.get_mean_deque()

    def run(self, dataset_info: dict, architecture: str, model_parameters: dict, optim: str, optim_kwargs: dict,
            epochs: int, scheduler: str, exec_str: str, save_metrics_n: int, checkpoint_n: int, evaluate_n: int,
            batch_size: int, valid_batch_size: int, shuffle: bool, num_workers: int, drop_last: bool,
            do_compile: bool, train_dataset: Dataset, val_dataset: Dataset, get_random_pseudo_experiment: Callable,
            run_name: str, run_dir: str, device: str, x_metrics: dict | None, y_metrics: dict | None) -> DirData:

        print(f"\n###\n\nTraining model {run_name} with {architecture} architecture\n\nmodel_parameters: {model_parameters}\n\n###\n")
        torch.set_float32_matmul_precision('high')
        param_labels = dataset_info['param_labels']
        x_metrics = get_metrics(x_metrics)
        y_metrics = get_metrics(y_metrics)

        data: DirData = self.get_data_object()
        print(f'Base dir: {data.dir}')
        with open(data.dir / f'model_parameters.json', 'w') as f:
            json.dump(model_parameters, f)

        architecture_class = eval(f'models.{architecture}')
        torch.manual_seed(1)
        np.random.seed(1)

        runs_dir = RUNS_DIR / run_dir / run_name
        runs_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(runs_dir)

        reporter = Reporter(
            hyper_parameters=model_parameters,
            deque_max_len=save_metrics_n,
            batch_size=batch_size,
        )

        net = architecture_class(**model_parameters).to(device)
        if do_compile:
            net = torch.compile(net)

        loss_func = nn.MSELoss()
        opt = eval(f"torch.optim.{optim}")(net.parameters(), **optim_kwargs)

        train_ldr = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, pin_memory=True)
        scheduler = eval(scheduler)

        try:
            total_batch_id = 0
            for epoch_id in range(epochs):
                batch_iterator = tqdm(train_ldr, ncols=120, position=0)
                for batch_id, (batch_x, batch_y, epsp) in enumerate(batch_iterator):
                    x_true = batch_x.to(device)
                    y_true = batch_y.to(device)
                    y_pred = net(x_true)

                    loss_val = loss_func(y_pred, y_true)  # [bs,12,9] [bs,9]

                    opt.zero_grad()
                    loss_val.backward()  # compute gradients

                    reporter.add_deque('Loss/train', loss_val.item(), is_batched=True)

                    with torch.no_grad():
                        for metric_name, metric_func in y_metrics.items():
                            reporter.add_deque(metric_name, metric_func(y_pred=y_pred, y_true=y_true), is_batched=True)

                        if param_labels is not None:
                            for param_value, param_name in zip(torch.mean((y_pred - y_true) ** 2, dim=0).cpu().numpy().tolist(), param_labels):
                                reporter.add_deque(f'{param_name}/train', param_value, is_batched=True)

                        # comparing x_pred and x_true
                        for sample_id, (x_i_true, y_i_pred_npy, epsp_i) in enumerate(
                                zip(x_true.detach().cpu(), y_pred.detach().cpu().numpy(), epsp.cpu().numpy())):
                            x_i_pred = torch.unsqueeze(torch.from_numpy(get_random_pseudo_experiment(scaled_params=y_i_pred_npy, experiment=Experiment(epsp_i))[0]),
                                                       0)
                            for metric_name, metric_func in x_metrics.items():
                                reporter.add_deque(metric_name, metric_func(x_pred=x_i_pred, x_true=x_i_true), is_batched=False)

                        if save_metrics_n and total_batch_id % save_metrics_n == save_metrics_n - 1:

                            reporter.add_scalar('LR', scheduler.get_last_lr()[-1], total_batch_id)
                            writer.add_scalar('LR', scheduler.get_last_lr()[-1], total_batch_id)
                            writer.add_scalar('Loss/train/_all', loss_val.item(), total_batch_id)
                            if param_labels is not None:
                                for param_name in param_labels:
                                    writer.add_scalar(f'{param_name}/train', reporter.get_mean_deque(f'{param_name}/train'), total_batch_id)
                                    reporter.add_scalar(f'{param_name}/train', reporter.get_mean_deque(f'{param_name}/train'), total_batch_id)

                            for metric_name, metric_func in chain(x_metrics.items(), y_metrics.items()):
                                writer.add_scalar(f'{metric_name}/train', reporter.get_mean_deque(metric_name), total_batch_id)
                                reporter.add_scalar(f'{metric_name}/train', reporter.get_mean_deque(metric_name), total_batch_id)

                        if evaluate_n and total_batch_id % evaluate_n == evaluate_n - 1:
                            for metric_name, metric_val in self.evaluate_model(valid_batch_size, dataset_info, x_metrics, y_metrics, net, val_dataset, device,
                                           num_workers, drop_last, get_random_pseudo_experiment).items():
                                writer.add_scalar(metric_name, metric_val, total_batch_id)
                                reporter.add_scalar(metric_name, metric_val, total_batch_id)

                    if checkpoint_n and total_batch_id % checkpoint_n == checkpoint_n - 1:
                        torch.save(net.state_dict(), data.dir / f'weights_{total_batch_id}.pt')
                    if exec_str is not None:
                        exec(exec_str)  # execute some code after gradients are computed
                    opt.step()  # update weights

                    batch_iterator.set_description(f'E{epoch_id+1}/{epochs}, LR: {scheduler.get_last_lr()[-1]: .5f}, Loss: {reporter.get_mean_deque("Loss/train"): .5f}')
                    total_batch_id += 1
                    scheduler.step()

            torch.save(net.state_dict(), data.dir / f'weights.pt')
            reporter.save(runs_dir / f'stats.pickle')
        except KeyboardInterrupt:
            return data
        return data


class TrainedModelTask(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = [
            TrainModelTask,
        ]
        parameters = [
            Parameter("architecture"),
            Parameter("model_parameters"),
            Parameter("chosen_checkpoint", default=None),
            Parameter("do_compile", default=True, ignore_persistence=True),
            Parameter("device", default="cuda", ignore_persistence=True),
        ]

    def run(self, train_model: Path, architecture: str, model_parameters: dict, do_compile: bool, device: str,
            chosen_checkpoint: int | None) -> nn.Module:
        if chosen_checkpoint is not None:
            weights_path = train_model / f'weights_{chosen_checkpoint}.pt'
        else:
            weights_path = train_model / 'weights.pt'
        architecture_class = eval(f'models.{architecture}')
        net = architecture_class(**model_parameters)
        net = net.to(device)
        if do_compile:
            net = torch.compile(net)

        net.load_state_dict(torch.load(str(weights_path)))
        return net


class ModelInfoTask(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = []
        parameters = [
            Parameter("architecture"),
            Parameter("model_parameters"),
            Parameter("run_name", ignore_persistence=True),
        ]

    def run(self, architecture: str, model_parameters: dict, run_name: str) -> dict:
        return {
            'architecture': architecture,
            'model_parameters': model_parameters,
            'run_name': run_name,
        }
