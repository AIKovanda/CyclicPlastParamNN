import json
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from taskchain import Parameter, Task, DirData
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rcpl.config import RUNS_DIR
from rcpl.rcpl import Experiment
from rcpl.reporter import Reporter
from rcpl.tasks.dataset import TrainDataset, ValDataset, GetRandomPseudoExperiment

from functools import partial
from rcpl import metrics, models


def get_metrics(metrics_config: dict | None):
    if metrics_config is None:
        return {}
    return {key: eval(value) for key, value in metrics_config.items()}


class TrainModel(Task):
    class Meta:
        input_tasks = [
            TrainDataset,
            ValDataset,
            GetRandomPseudoExperiment,
        ]
        parameters = [
            Parameter("architecture"),
            Parameter("model_parameters"),
            Parameter("optim", default='AdamW'),
            Parameter("optim_kwargs"),
            Parameter("epochs"),
            Parameter("scheduler", default='None'),
            Parameter("exec_str", default=None),
            Parameter("validate_batch_n", default=100, ignore_persistence=True),
            Parameter("batch_size"),
            Parameter("shuffle", default=True),
            Parameter("num_workers", default=-1, ignore_persistence=True),
            Parameter("drop_last", default=True),
            Parameter("do_compile", default=True, ignore_persistence=True),
            Parameter("run_name", ignore_persistence=True),
            Parameter("device", default="cuda", ignore_persistence=True),
            Parameter("x_metrics", default=None, ignore_persistence=True),
            Parameter("y_metrics", default=None, ignore_persistence=True),
            Parameter("deque_x_metrics", default=None, ignore_persistence=True),
            Parameter("deque_y_metrics", default=None, ignore_persistence=True),
        ]

    def run(self, architecture: str, model_parameters: dict, optim: str, optim_kwargs: dict, epochs: int, scheduler,
            exec_str: str, validate_batch_n: int, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool,
            do_compile: bool, train_dataset: Dataset, val_dataset: Dataset, get_random_pseudo_experiment: Callable,
            run_name: str, device: str, x_metrics: dict | None, y_metrics: dict | None, deque_x_metrics: dict | None,
            deque_y_metrics: dict | None) -> DirData:

        x_metrics = get_metrics(x_metrics)
        y_metrics = get_metrics(y_metrics)
        deque_x_metrics = get_metrics(deque_x_metrics)
        deque_y_metrics = get_metrics(deque_y_metrics)

        data: DirData = self.get_data_object()

        architecture_class = eval(f'models.{architecture}')
        torch.manual_seed(1)
        np.random.seed(1)

        runs_dir = RUNS_DIR / run_name
        runs_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(runs_dir)

        reporter = Reporter(
            hyper_parameters=model_parameters,
            deque_x_max_len=100 * batch_size,
            deque_y_max_len=100,
        )

        net = architecture_class(**model_parameters).to(device)
        if do_compile:
            print('-- compilation started')
            net = torch.compile(net)
            print('-- compilation finished')

        loss_func = nn.MSELoss()
        opt = eval(f"torch.optim.{optim}")(net.parameters(), **optim_kwargs)

        train_ldr = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
        batch_iterator = tqdm(train_ldr, ncols=150)
        scheduler = eval(scheduler)

        total_batch_id = 0
        for epoch_id in range(epochs):
            print(f"### Epoch {epoch_id + 1}/{epochs} ###")
            for batch_id, (batch_x, batch_y, epsp) in enumerate(batch_iterator):
                x_true = batch_x.to(device)
                y_true = batch_y.to(device)
                y_pred = net(x_true)

                loss_val = loss_func(y_pred, y_true)  # [bs,12,9] [bs,9]
                reporter.add_deque('Loss/train', loss_val.item(), 'y')

                with torch.no_grad():
                    for metric_name, metric_func in deque_y_metrics.items():
                        reporter.add_deque(metric_name, metric_func(y_pred=y_pred, y_true=y_true), 'y')

                    if validate_batch_n and total_batch_id % validate_batch_n == validate_batch_n - 1:
                        reporter.add_scalar('LR', scheduler.get_last_lr()[-1], total_batch_id)
                        writer.add_scalar('LR', scheduler.get_last_lr()[-1], total_batch_id)
                        writer.add_scalar('Loss/train', loss_val.item(), total_batch_id)
                        print(torch.mean((y_pred - y_true) ** 2, dim=0))
                        for metric_name, metric_func in y_metrics.items():
                            reporter.add_scalar(metric_name, metric_func(y_pred=y_pred, y_true=y_true), 'y')

                    # comparing x_pred and x_true
                    for sample_id, (x_i_true, y_i_pred_npy, epsp_i) in enumerate(
                            zip(x_true.detach().cpu(), y_pred.detach().cpu().numpy(), epsp.cpu().numpy())):
                        x_i_pred = torch.unsqueeze(torch.from_numpy(get_random_pseudo_experiment(scaled_params=y_i_pred_npy, experiment=Experiment(epsp_i))[0]),
                                                   0)  # exp.get_from_scaled_params(y_i_pred_npy).data_arr[0, :]
                        # if training_params.get('validate_batch_n') % 1000 == 999 and sample_id == 0:
                        #     fig = plt.figure(figsize=(16,9))
                        #     plt.plot(np.insert(x_i_true_npy,0,0), label='ground true')
                        #     plt.plot(np.insert(x_i_pred_npy,0,0), label='predicted')
                        #     plt.legend()
                        #     plt.savefig(stat_dir / f'batch{total_batch_id}.png')
                        #     writer.add_figure('Predicted_vs_True', fig, total_batch_id)
                        #     plt.close()

                        for metric_name, metric_func in deque_x_metrics.items():
                            reporter.add_deque(metric_name, metric_func(x_pred=x_i_pred, x_true=x_i_true), 'x')

                        if validate_batch_n and total_batch_id % validate_batch_n == validate_batch_n - 1:
                            for metric_name, metric_func in x_metrics.items():
                                reporter.add_scalar(metric_name, metric_func(x_pred=x_i_pred, x_true=x_i_true),
                                                    f'{total_batch_id}_{sample_id}')

                opt.zero_grad()
                loss_val.backward()  # compute gradients
                if exec_str is not None:
                    exec(exec_str)  # execute some code after gradients are computed
                opt.step()  # update weights

                batch_iterator.set_description(f'LR: {scheduler.get_last_lr()[-1]: .5f}, {reporter.report()}')

                # comparing to reference
                # net.eval()
                # for sample_id, (x_i_npy, y_i_pred_npy) in enumerate(zip(reference_x_true.detach().cpu().numpy(), net(reference_x_true).detach().cpu().numpy())):
                #     x_i_pred = get_random_pseudo_experiment(scaled_params=y_i_pred_npy, **dataset_config['rcpl_params'])[0]  # exp.get_from_scaled_params(y_i_pred_npy).data_arr[0, :]
                #     x_i_true_npy = x_i_npy.T.reshape((-1))
                #     fig = plt.figure(figsize=(16,9))
                #     plt.plot(np.insert(x_i_true_npy,0,0), label='ground true')
                #     plt.plot(np.insert(x_i_pred, 0, 0), label='predicted')
                #     plt.legend()
                #     writer.add_figure(f'Reference/{sample_id}', fig, total_batch_id)
                #     plt.close()
                #
                # net.train()

                total_batch_id += 1
                scheduler.step()

        torch.save(net.state_dict(), data.dir / f'weights.pt')
        with open(data.dir / f'model_parameters.json', 'w') as f:
            json.dump(model_parameters, f)
        reporter.save(data.dir / f'stats.pickle')
        return data
