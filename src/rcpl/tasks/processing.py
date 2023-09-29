import json
from itertools import chain
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from taskchain import Parameter, Task, DirData, InMemoryData
from torch.utils.data import DataLoader, Subset
from scipy.optimize import fmin
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm, trange

from rcpl.config import RUNS_DIR, REAL_EXPERIMENT
from rcpl.rcpl import Experiment, rcpl_torch_batch, rcpl_torch_one, random_cyclic_plastic_loading
from rcpl.reporter import Reporter
from rcpl.tasks.dataset import TrainDatasetTask, ValDatasetTask, GetRandomPseudoExperimentTask, DatasetInfoTask, \
    UnscaleParamsTorchTask, GetCrlbTask, TestDatasetTask, ScaleParamsTorchTask

from functools import partial
from rcpl import metrics, models


# todo: put this to taskchain somehow
lower_bound = np.array([15, 100, 0.006666666666666667, 1000, 50, 50, 50, 0.000000001, 0.000000001, 0.000000001, 0.000000001])  # Grid lower bound (number or list of len d)
upper_bound = np.array([250, 10000, 0.03333333333333333, 10000, 2000, 2000, 2000, 350, 350, 350, 350])


def to_optimize_one(theta_, stress_, epsp_, kappa_dim, dim) -> float:
    theta_ = np.clip(theta_, a_min=lower_bound, a_max=upper_bound)
    res = random_cyclic_plastic_loading(
        k0=theta_[:1],
        kap=theta_[1:1 + kappa_dim],
        c=theta_[1 + kappa_dim:1 + kappa_dim + dim],
        a=theta_[1 + kappa_dim + dim:],
        epsp=epsp_,
    )
    return float(np.mean((res - stress_) ** 2))


def simplex(unscaled_theta: np.ndarray, stress: np.ndarray, kappa_dim: int, dim: int, epsp: np.ndarray) -> tuple[
    np.ndarray, list, list]:
    opt_theta = np.zeros_like(unscaled_theta)
    new_values = []
    old_values = []
    for i, (theta_i, stress_i, epsp_i) in enumerate(zip(unscaled_theta, stress, epsp)):
        now_value = to_optimize_one(theta_i, stress_i, epsp_i, kappa_dim, dim)
        # print('Now:', now_value)
        new_theta = fmin(partial(to_optimize_one, stress_=stress_i, epsp_=epsp_i, kappa_dim=kappa_dim, dim=dim),
                         theta_i, disp=0)

        new_theta = np.clip(new_theta, a_min=lower_bound, a_max=upper_bound)
        opt_theta[i] = new_theta
        new_value = to_optimize_one(opt_theta[i], stress_i, epsp_i, kappa_dim, dim)
        if now_value < new_value:
            opt_theta[i] = theta_i
            print('BUG - new is worse')
        # print('New:', new_value)
        # print('Old:', now_value)
        new_values.append(new_value)
        old_values.append(now_value)
    return opt_theta, new_values, old_values


def evaluate_model(valid_batch_size, dataset_info, x_metrics, y_metrics, net, val_dataset, device,
                   num_workers, drop_last, get_random_pseudo_experiment, use_simplex=False, use_tqdm=False,
                   tqdm_message=None, unscale_params_torch=None, scale_params_torch=None, prediction_is_scaled=True):
    net.eval()

    param_labels = dataset_info['param_labels']
    train_ldr = DataLoader(val_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=num_workers,
                           drop_last=drop_last, pin_memory=True)

    reporter = Reporter(batch_size=valid_batch_size)
    loss_func = nn.MSELoss()
    if use_tqdm:
        train_ldr = tqdm(train_ldr, ncols=120, desc=tqdm_message)
    for batch_id, (batch_x, batch_y, *epsp) in enumerate(train_ldr):
        x_true = batch_x.to(device)
        y_true = batch_y.to(device)
        with torch.no_grad():
            # comparing x_pred and x_true
            if len(epsp) == 0:
                epsp = x_true[:, 1, :]
            else:
                epsp = epsp[0]  # because *epsp gives a list of one element

            y_pred_raw = net(x_true)
            if use_simplex:
                if prediction_is_scaled:
                    unscaled_y_pred = unscale_params_torch(y_pred_raw)
                else:
                    unscaled_y_pred = y_pred_raw

                unscaled_y_pred, new_values, old_values = simplex(
                    unscaled_theta=unscaled_y_pred.cpu().numpy(),
                    stress=x_true[:, 0, :].cpu().numpy(),
                    kappa_dim=dataset_info['kappa_dim'],
                    dim=dataset_info['dim'],
                    epsp=epsp.cpu().numpy(),
                )
                unscaled_y_pred = torch.from_numpy(unscaled_y_pred).to(device)
                reporter.add_deque('old_x_l2', np.mean(old_values), use_max_len=False, do_print=False)
                reporter.add_deque('new_x_l2', np.mean(new_values), use_max_len=False, do_print=False)
                y_pred_scaled = scale_params_torch(unscaled_y_pred)

            else:
                if prediction_is_scaled:
                    y_pred_scaled = y_pred_raw
                else:
                    y_pred_scaled = scale_params_torch(y_pred_raw)
            reporter.add_deque('Loss/val/_all', loss_func(y_pred_scaled, y_true).item(), use_max_len=False, do_print=False)
            for metric_name, metric_func in y_metrics.items():
                reporter.add_deque(f'{metric_name}/val', metric_func(y_pred=y_pred_scaled, y_true=y_true), use_max_len=False)

            if param_labels is not None:
                for param_value, param_name in zip(torch.mean((y_pred_scaled - y_true) ** 2, dim=0).cpu().numpy().tolist(),
                                                   param_labels):
                    reporter.add_deque(f'{param_name}/val', param_value, use_max_len=False)

            for sample_id, (x_i_true, y_i_pred_npy, epsp_i) in enumerate(
                    zip(x_true.detach().cpu(), y_pred_scaled.detach().cpu().numpy(), epsp.cpu().numpy())):
                x_i_pred = torch.unsqueeze(torch.from_numpy(
                    get_random_pseudo_experiment(scaled_params=y_i_pred_npy, experiment=Experiment(epsp_i))[0]), 0)
                for metric_name, metric_func in x_metrics.items():
                    reporter.add_deque(f'{metric_name}/val', metric_func(x_pred=x_i_pred[:1], x_true=x_i_true[:1]),
                                       use_max_len=False, do_print=False)
    net.train()
    return reporter.get_mean_deque()


class RealExperimentTask(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = [
        ]
        parameters = [
            Parameter("takes_length"),
            Parameter("takes_epsp"),
        ]

    def run(self, takes_length, takes_epsp) -> dict[str, np.ndarray]:
        sigs = {}
        for exp_name, exp_config in REAL_EXPERIMENT.items():
            df_exp = pd.read_csv(exp_config['path'])
            sig = df_exp[exp_config['signal_col']].to_numpy()
            if takes_length is not None:
                sig = sig[:takes_length]
            if takes_epsp:
                sig = np.vstack([sig, np.array([exp_config['epsp']])])
            else:
                sig = np.expand_dims(sig, 0)
            sigs[exp_name] = sig

        return sigs


def get_metrics(metrics_config: dict | None):
    if metrics_config is None:
        return {}
    return {key: eval(value) for key, value in metrics_config.items()}


def contains_nan_gradients(model) -> bool:
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                return True
    return False


class LossFuncTask(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = [UnscaleParamsTorchTask, DatasetInfoTask]
        parameters = [
            Parameter("loss_func", default='MSELoss'),
            Parameter("loss_func_kwargs", default=None),
        ]

    def run(self, loss_func: str, loss_func_kwargs: dict, unscale_params_torch, dataset_info: dict) -> Callable:
        if loss_func_kwargs is None:
            loss_func_kwargs = {}
        if loss_func == 'MSELoss':
            def loss_func(y_pred, y_true, x_true, epsp):
                return F.mse_loss(y_pred, y_true, reduction="mean")

            return loss_func

        elif loss_func == 'epsp':
            def loss_func(y_pred, y_true, x_true, epsp):
                # unscale
                y_pred_unscaled = unscale_params_torch(y_pred)
                x_pred = rcpl_torch_batch(theta=y_pred_unscaled, epsp=epsp.float().to(y_pred_unscaled.device),
                                          dim=dataset_info['dim'], kappa_dim=dataset_info['kappa_dim'])
                # normalize
                mse_loss = F.mse_loss(
                    x_pred[:, :] / dataset_info['x_std'],
                    x_true[:, 0, :] / dataset_info['x_std'],
                    reduction="mean") * loss_func_kwargs.get('x_grad_scale', 1)
                if (y_ratio := loss_func_kwargs.get('y_ratio', 0)) > 0:
                    mse_loss = (1 - y_ratio) * mse_loss + y_ratio * F.mse_loss(y_pred, y_true, reduction="mean")
                return mse_loss

            return loss_func
        else:
            raise ValueError(f'Unknown loss function: {loss_func}')


class TrainModelTask(Task):
    class Meta:
        input_tasks = [
            DatasetInfoTask,
            TrainDatasetTask,
            ValDatasetTask,
            GetRandomPseudoExperimentTask,
            LossFuncTask,
        ]
        parameters = [
            Parameter("architecture"),
            Parameter("model_parameters"),
            Parameter("optim", default='AdamW'),
            Parameter("optim_kwargs", default=None),
            Parameter("epochs", default=1),
            Parameter("scheduler", default='None'),
            Parameter("exec_str", default=None),
            Parameter("save_metrics_n", default=100, ignore_persistence=True),
            Parameter("evaluate_n", default=300, ignore_persistence=True),
            Parameter("checkpoint_n", default=2000),
            Parameter("batch_size", default=32),
            Parameter("is_trainable", default=True, dont_persist_default_value=True),
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

    def run(self, dataset_info: dict, architecture: str, model_parameters: dict, optim: str, optim_kwargs: dict,
            epochs: int, scheduler: str, exec_str: str, save_metrics_n: int, checkpoint_n: int, evaluate_n: int,
            batch_size: int, valid_batch_size: int, shuffle: bool, num_workers: int, drop_last: bool,
            do_compile: bool, train_dataset: Dataset, val_dataset: Dataset, get_random_pseudo_experiment: Callable,
            run_name: str, run_dir: str, device: str, x_metrics: dict | None, y_metrics: dict | None,
            loss_func: callable, is_trainable: bool) -> DirData:

        data: DirData = self.get_data_object()
        if not is_trainable:
            return data
        print(
            f"\n###\n\nTraining model {run_name} with {architecture} architecture\n\nmodel_parameters: {model_parameters}\n\n###\n")
        torch.set_float32_matmul_precision('high')
        param_labels = dataset_info['param_labels']
        x_metrics = get_metrics(x_metrics)
        y_metrics = get_metrics(y_metrics)

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

        opt = eval(f"torch.optim.{optim}")(net.parameters(), **optim_kwargs)

        train_ldr = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                               drop_last=drop_last, pin_memory=True)
        scheduler = eval(scheduler)

        try:
            total_batch_id = 0
            for epoch_id in range(epochs):
                batch_iterator = tqdm(train_ldr, ncols=120, position=0)
                for batch_id, (batch_x, batch_y, *epsp) in enumerate(batch_iterator):
                    x_true = batch_x.to(device)
                    y_true = batch_y.to(device)
                    y_pred = net(x_true)
                    # y_pred.retain_grad()

                    if len(epsp) == 0:
                        epsp = x_true[:, 1, :]
                    else:
                        epsp = epsp[0]

                    loss_val = loss_func(y_pred, y_true, x_true, epsp)  # [bs,12,9] [bs,9]

                    opt.zero_grad()
                    loss_val.backward()  # compute gradients
                    # print("šřěěšřč grad:", y_pred.grad.mean(), y_pred.grad.std())

                    reporter.add_deque('Loss/train', loss_val.item(), is_batched=True)

                    with torch.no_grad():
                        for metric_name, metric_func in y_metrics.items():
                            reporter.add_deque(metric_name, metric_func(y_pred=y_pred, y_true=y_true), is_batched=True)

                        if param_labels is not None:
                            for param_value, param_name in zip(
                                    torch.mean((y_pred - y_true) ** 2, dim=0).cpu().numpy().tolist(), param_labels):
                                reporter.add_deque(f'{param_name}/train', param_value, is_batched=True)

                        # comparing x_pred and x_true
                        for sample_id, (x_i_true, y_i_pred_npy, epsp_i) in enumerate(
                                zip(x_true.detach().cpu(), y_pred.detach().cpu().numpy(), epsp.cpu().numpy())):
                            x_i_pred = torch.unsqueeze(torch.from_numpy(
                                get_random_pseudo_experiment(scaled_params=y_i_pred_npy, experiment=Experiment(epsp_i))[
                                    0]),
                                0)
                            for metric_name, metric_func in x_metrics.items():
                                reporter.add_deque(metric_name, metric_func(x_pred=x_i_pred[:1], x_true=x_i_true[:1]),
                                                   is_batched=False)

                        if save_metrics_n and total_batch_id % save_metrics_n == save_metrics_n - 1:

                            reporter.add_scalar('LR', scheduler.get_last_lr()[-1], total_batch_id)
                            writer.add_scalar('LR', scheduler.get_last_lr()[-1], total_batch_id)
                            writer.add_scalar('Loss/train/_all', loss_val.item(), total_batch_id)
                            if param_labels is not None:
                                for param_name in param_labels:
                                    writer.add_scalar(f'{param_name}/train',
                                                      reporter.get_mean_deque(f'{param_name}/train'), total_batch_id)
                                    reporter.add_scalar(f'{param_name}/train',
                                                        reporter.get_mean_deque(f'{param_name}/train'), total_batch_id)

                            for metric_name, metric_func in chain(x_metrics.items(), y_metrics.items()):
                                writer.add_scalar(f'{metric_name}/train', reporter.get_mean_deque(metric_name),
                                                  total_batch_id)
                                reporter.add_scalar(f'{metric_name}/train', reporter.get_mean_deque(metric_name),
                                                    total_batch_id)

                        if evaluate_n and total_batch_id % evaluate_n == evaluate_n - 1:
                            for metric_name, metric_val in evaluate_model(valid_batch_size, dataset_info, x_metrics,
                                                                          y_metrics, net, val_dataset, device,
                                                                          num_workers, drop_last,
                                                                          get_random_pseudo_experiment).items():
                                writer.add_scalar(metric_name, metric_val, total_batch_id)
                                reporter.add_scalar(metric_name, metric_val, total_batch_id)

                    if checkpoint_n and total_batch_id % checkpoint_n == checkpoint_n - 1:
                        torch.save(net.state_dict(), data.dir / f'weights_{total_batch_id}.pt')
                    if exec_str is not None:
                        exec(exec_str)  # execute some code after gradients are computed
                    if contains_nan_gradients(net):
                        print("NaN gradients detected. Skipping step.")
                    else:
                        opt.step()  # update weights

                    batch_iterator.set_description(
                        f'E{epoch_id + 1}/{epochs}, LR: {scheduler.get_last_lr()[-1]: .5f}, Loss: {reporter.get_mean_deque("Loss/train"): .5f}')
                    total_batch_id += 1
                    scheduler.step()

            torch.save(net.state_dict(), data.dir / f'weights.pt')
            reporter.save(runs_dir / f'stats.pickle')
        except KeyboardInterrupt:
            return data
        return data


class ChooseNet:
    def __init__(self, architecture: str, model_parameters: dict, do_compile: bool, device: str, train_model: Path):
        self.architecture = architecture
        self.model_parameters = model_parameters
        self.do_compile = do_compile
        self.device = device
        self.train_model = train_model

    @property
    def available_checkpoints(self):
        return sorted([int(str(f).split('_')[-1].split('.')[0]) for f in self.train_model.glob('weights_*.pt')])

    def __call__(self, chosen_checkpoint: int | None):
        if chosen_checkpoint is not None:
            weights_path = self.train_model / f'weights_{chosen_checkpoint}.pt'
        else:
            weights_path = self.train_model / 'weights.pt'
        architecture_class = eval(f'models.{self.architecture}')
        net = architecture_class(**self.model_parameters)
        net = net.to(self.device)
        if self.do_compile:
            net = torch.compile(net)

        net.load_state_dict(torch.load(str(weights_path)))
        return net


class ChooseModelTask(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = [
            TrainModelTask,
        ]
        parameters = [
            Parameter("architecture"),
            Parameter("model_parameters"),
            Parameter("do_compile", default=True, ignore_persistence=True),
            Parameter("device", default="cuda", ignore_persistence=True),
        ]

    def run(self, train_model: Path, architecture: str, model_parameters: dict, do_compile: bool, device: str,
            ) -> ChooseNet:
        cn = ChooseNet(architecture, model_parameters, do_compile, device, train_model)
        return cn


class TrainedModelTask(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = [
            TrainModelTask,
        ]
        parameters = [
            Parameter("architecture"),
            Parameter("model_parameters"),
            Parameter("is_trainable", default=True, dont_persist_default_value=True),
            Parameter("chosen_checkpoint", default=None),
            Parameter("do_compile", default=True, ignore_persistence=True),
            Parameter("device", default="cuda", ignore_persistence=True),
        ]

    def run(self, train_model: Path, architecture: str, model_parameters: dict, do_compile: bool, device: str,
            is_trainable: bool,
            chosen_checkpoint: int | None) -> object:
        if chosen_checkpoint is not None:
            weights_path = train_model / f'weights_{chosen_checkpoint}.pt'
        else:
            weights_path = train_model / 'weights.pt'
        architecture_class = eval(f'models.{architecture}')
        net = architecture_class(**model_parameters)
        net = net.to(device)
        if do_compile:
            net = torch.compile(net)
        if is_trainable:
            net.load_state_dict(torch.load(str(weights_path)))
        return net


class ModelInfoTask(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = []
        parameters = [
            Parameter("architecture"),
            Parameter("model_parameters"),
            Parameter("run_name"),
            Parameter("run_dir"),
            Parameter("takes_epsp"),
        ]

    def run(self, architecture: str, model_parameters: dict, run_name: str, run_dir: str, takes_epsp: bool) -> dict:
        return {
            'architecture': architecture,
            'model_parameters': model_parameters,
            'run_name': run_name,
            'run_dir': run_dir,
            'takes_epsp': takes_epsp,
        }


class ValidateCrlbTask(Task):
    class Meta:
        input_tasks = [RealExperimentTask, GetCrlbTask, TrainedModelTask, UnscaleParamsTorchTask, DatasetInfoTask,
                       ModelInfoTask]
        parameters = [
            Parameter("device", default="cuda", ignore_persistence=True),
            Parameter("crlb_runs", default=1000),
        ]

    def run(self, real_experiment, get_crlb, trained_model, device, unscale_params_torch, crlb_runs,
            dataset_info, model_info) -> dict:
        out = {}
        trained_model.eval()
        all_unscaled_params = torch.zeros((crlb_runs, 1 + 2 * dataset_info['dim'] + dataset_info['kappa_dim']))
        for experiment_name, sig in real_experiment.items():
            with torch.no_grad():
                theta_hat = trained_model(torch.unsqueeze(torch.from_numpy(sig), 0).float().to(device))[0]
            epsp = REAL_EXPERIMENT[experiment_name]['epsp']
            unscaled_params = unscale_params_torch(theta_hat)
            crlb = get_crlb(epsp, unscaled_params)
            with torch.no_grad():
                for i in trange(crlb_runs):
                    # uniform random increment
                    random_increment = torch.randn_like(theta_hat) * torch.from_numpy(crlb['std']).to(device) / 10000
                    unscaled_params_mod = torch.clip(unscaled_params + random_increment, min=1e-9, max=None)
                    sig_hat = rcpl_torch_one(theta=unscaled_params_mod, epsp=torch.tensor(epsp).to(device),
                                             dim=dataset_info['dim'], kappa_dim=dataset_info['kappa_dim'])
                    if model_info['takes_epsp']:
                        sig_hat = torch.unsqueeze(torch.vstack([sig_hat, torch.tensor([epsp]).to(device)]), 0).float()
                    else:
                        sig_hat = torch.unsqueeze(torch.unsqueeze(sig_hat.float(), 0), 0).to(device)
                    theta_hat_hat = trained_model(sig_hat)[0]
                    all_unscaled_params[i] = (unscale_params_torch(theta_hat_hat) - random_increment).cpu()

            out[experiment_name] = {
                'theta_mean': torch.mean(all_unscaled_params, dim=0).numpy().tolist(),
                'theta_std': torch.std(all_unscaled_params, dim=0).numpy().tolist(),
                'theta_relative_std': (
                        torch.std(all_unscaled_params, dim=0) / unscaled_params.detach().cpu()).numpy().tolist(),
                'crlb': {key: val.tolist() for key, val in crlb.items()},
            }

        return out


class ModelMetricsTask(Task):
    class Meta:
        input_tasks = [
            TrainedModelTask,
            DatasetInfoTask,
            TestDatasetTask,
            GetRandomPseudoExperimentTask,
            ScaleParamsTorchTask,
            UnscaleParamsTorchTask
        ]
        parameters = [
            Parameter("eval_batch_size", default=1, ignore_persistence=True),
            Parameter("num_workers", default=-1, ignore_persistence=True),
            Parameter("drop_last", default=True),
            Parameter("run_dir", ignore_persistence=True),
            Parameter("run_name", ignore_persistence=True),
            Parameter("device", default="cuda", ignore_persistence=True),
            Parameter("x_metrics", default=None, ignore_persistence=True),
            Parameter("y_metrics", default=None, ignore_persistence=True),
            Parameter("prediction_is_scaled", default=True, dont_persist_default_value=True),
            Parameter("evaluate_limit", default=1024),
        ]

    def run(self, trained_model, dataset_info: dict, eval_batch_size: int, num_workers: int, drop_last: bool,
            test_dataset: Dataset, get_random_pseudo_experiment: Callable, evaluate_limit: int,
            run_name: str, run_dir: str, device: str, x_metrics: dict | None, y_metrics: dict | None,
            scale_params_torch, unscale_params_torch, prediction_is_scaled) -> dict:
        x_metrics = get_metrics(x_metrics)
        y_metrics = get_metrics(y_metrics)
        runs_dir = RUNS_DIR / run_dir / run_name
        runs_dir.mkdir(parents=True, exist_ok=True)
        trained_model.eval()

        dataset = Subset(test_dataset, indices=range(evaluate_limit)) if evaluate_limit > 0 else test_dataset
        metric_results = {}
        for dataset_type, use_simplex in zip(['simplex', 'raw'], [True, False]):
            metric_results[dataset_type] = {
                key: float(val) for key, val in
                evaluate_model(eval_batch_size, dataset_info, x_metrics, y_metrics, trained_model, dataset, device,
                               num_workers, drop_last, get_random_pseudo_experiment, use_simplex=use_simplex,
                               use_tqdm=True, tqdm_message=f'{dataset_type}', unscale_params_torch=unscale_params_torch,
                               scale_params_torch=scale_params_torch,
                               prediction_is_scaled=prediction_is_scaled).items()}
            print(f'{dataset_type} metrics: {metric_results[dataset_type]}')
        with open(runs_dir / f'model_metrics.json', 'w') as f:
            json.dump(metric_results, f)
        return metric_results
