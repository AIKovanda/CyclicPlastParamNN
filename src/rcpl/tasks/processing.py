import json
from datetime import datetime
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import fmin
from taskchain import Parameter, ModuleTask, DirData, InMemoryData
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from rcpl import metrics
from rcpl.config import RUNS_DIR, MEASURED_EXP_DIR
from rcpl.cpl_models import CPLModelFactory
from rcpl.experiment import Experiment
from rcpl.reporter import Reporter
from rcpl.tasks.dataset import TrainDatasetTask, ValDatasetTask, GetRandomPseudoExperimentTask, DatasetInfoTask, \
    UnscaleThetaTorchTask, GetCrlbTask, TestDatasetTask, ScaleThetaTorchTask, ModelFactoryTask


def validate(unscaled_theta, lower_bound, upper_bound):
    if np.any(unscaled_theta < lower_bound-1e-9):
        return False
    if np.any(unscaled_theta > upper_bound+1e-9):
        return False
    return True


def to_optimize_one(unscaled_theta, true_stress, signal_, model_factory: CPLModelFactory) -> float:
    if not validate(unscaled_theta, model_factory.simplex_lower_bound, model_factory.simplex_upper_bound):
        return np.inf
    res = model_factory.make_model(theta=unscaled_theta).predict_stress(signal_)
    return float(np.mean((res - true_stress) ** 2))


def simplex(unscaled_theta: np.ndarray, true_stress: np.ndarray, signal: np.ndarray, model_factory: CPLModelFactory, verbose=True) -> tuple[
        np.ndarray, list, list]:
    opt_theta = np.zeros_like(unscaled_theta)
    new_values = []
    old_values = []
    for i, (theta_i, true_stress_i, signal_i) in enumerate(zip(unscaled_theta, true_stress, signal)):
        if not validate(theta_i, model_factory.simplex_lower_bound, model_factory.simplex_upper_bound):
            more_than_bound = theta_i - model_factory.simplex_upper_bound
            more_than_bound[more_than_bound < 0] = 0
            less_than_bound = model_factory.simplex_lower_bound - theta_i
            less_than_bound[less_than_bound < 0] = 0
            if verbose:
                print(f'BUG - not valid - clipping...\n'
                      f'Above bound: {more_than_bound}\n'
                      f'Below bound: {less_than_bound}')
            theta_i = np.clip(theta_i, model_factory.simplex_lower_bound+1e-9, model_factory.simplex_upper_bound-1e-9)
        assert validate(theta_i, model_factory.simplex_lower_bound, model_factory.simplex_upper_bound)
        now_value = to_optimize_one(theta_i, true_stress_i, signal_i, model_factory=model_factory)
        # print('Now:', now_value)
        new_theta_i = fmin(partial(to_optimize_one, true_stress=true_stress_i, signal_=signal_i, model_factory=model_factory),
                           theta_i, disp=0)

        assert validate(new_theta_i, model_factory.simplex_lower_bound, model_factory.simplex_upper_bound)
        opt_theta[i] = new_theta_i
        new_value = to_optimize_one(opt_theta[i], true_stress_i, signal_i, model_factory=model_factory)
        if now_value < new_value:
            opt_theta[i] = theta_i
            print('BUG - new is worse')
        # print('New:', new_value)
        # print('Old:', now_value)
        new_values.append(new_value)
        old_values.append(now_value)
    return opt_theta, new_values, old_values


def evaluate_model(valid_dataloader_kwargs, model_factory: CPLModelFactory, dataset_info, x_metrics, y_metrics, model,
                   val_dataset, device, get_random_pseudo_experiment, eval_type='raw', use_tqdm=False,
                   tqdm_message=None, unscale_theta_torch=None, scale_theta_torch=None, prediction_is_scaled=True):
    model.eval()

    labels = dataset_info['labels']
    train_ldr = DataLoader(val_dataset, **valid_dataloader_kwargs)

    reporters = [Reporter(), Reporter()]  # raw, simplex
    mse_loss_func = nn.MSELoss()
    if use_tqdm:
        train_ldr = tqdm(train_ldr, ncols=120, desc=tqdm_message)
    for batch_id, (batch_x, batch_y, *signal) in enumerate(train_ldr):
        x_true = batch_x.to(device)
        y_true = batch_y.to(device)
        with torch.no_grad():
            # comparing x_pred and x_true
            signal = x_true[:, 1:, :] if len(signal) == 0 else signal[0]  # because *signal gives a list of one element
            y_pred_raw = model(x_true)
            for reporter_id, use_simplex in enumerate([False, True]):
                if use_simplex and eval_type == 'raw':
                    continue
                if not use_simplex and eval_type == 'simplex':
                    continue

                if use_simplex:
                    if prediction_is_scaled:
                        unscaled_y_pred = unscale_theta_torch(y_pred_raw.clone())
                    else:
                        unscaled_y_pred = y_pred_raw.clone()

                    unscaled_y_pred, new_values, old_values = simplex(
                        unscaled_theta=unscaled_y_pred.cpu().numpy(),
                        true_stress=x_true[:, 0, :].cpu().numpy(),
                        signal=signal.cpu().numpy(),
                        model_factory=model_factory,
                        verbose=False,
                    )
                    unscaled_y_pred = torch.from_numpy(unscaled_y_pred).to(device)
                    reporters[reporter_id].add_mean('old_x_l2', np.mean(old_values))
                    reporters[reporter_id].add_mean('new_x_l2', np.mean(new_values))
                    y_pred_scaled = scale_theta_torch(unscaled_y_pred)

                else:
                    if prediction_is_scaled:
                        y_pred_scaled = y_pred_raw.clone()
                    else:
                        y_pred_scaled = scale_theta_torch(y_pred_raw.clone())
                
                reporters[reporter_id].add_mean('Loss/val/_all', mse_loss_func(y_pred_scaled, y_true).item())
                for metric_name, metric_func in y_metrics.items():
                    reporters[reporter_id].add_mean(f'{metric_name}/val', metric_func(y_pred=y_pred_scaled, y_true=y_true))

                if labels is not None:
                    for param_value, param_name in zip(torch.mean((y_pred_scaled - y_true) ** 2, dim=0).cpu().numpy().tolist(), labels):
                        reporters[reporter_id].add_mean(f'{param_name}/val', param_value)

                # comparing x_pred and x_true
                if len(x_metrics) > 0:
                    for item_id, (x_i_true, y_i_pred_npy, signal_i) in enumerate(zip(x_true.detach(),
                                  y_pred_scaled.detach().cpu().numpy(), signal.cpu().numpy())):
                        x_i_pred = torch.from_numpy(
                            get_random_pseudo_experiment(scaled_theta=y_i_pred_npy, signal_correct_representation=signal_i)).to(device)

                        for metric_name, metric_func in x_metrics.items():
                            reporters[reporter_id].add_mean(f'{metric_name}/val', metric_func(x_pred=x_i_pred, x_true=x_i_true[:1]))

    model.train()
    if eval_type == 'raw':
        return reporters[0].get_mean()
    elif eval_type == 'simplex':
        return reporters[1].get_mean()
    else:
        return {'raw': reporters[0].get_mean(), 'simplex': reporters[1].get_mean()}


def get_metrics(metrics_config: dict | None):
    if metrics_config is None:
        return {}
    return {key: getattr(metrics, value) for key, value in metrics_config.items()}


def contains_nan_gradients(model) -> bool:
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                return True
    return False


class RealExperimentTask(ModuleTask):
    class Meta:
        data_class = InMemoryData
        input_tasks = [
            DatasetInfoTask,
        ]
        parameters = [
            Parameter("takes_max_length"),
            Parameter("takes_channels"),
        ]

    def run(self, dataset_info: dict, takes_max_length: int | None, takes_channels: list[str]) -> Callable:
        def get_prepared_experiment(experiment_path: Path | str):
            if isinstance(experiment_path, str):
                if not experiment_path.endswith('.json'):
                    experiment_path += '.json'
                experiment_path = MEASURED_EXP_DIR / experiment_path
            experiment = Experiment(json_path=experiment_path)
            nn_input = experiment.get_signal_representation(dataset_info['exp_representation'], channels=takes_channels)
            stress = experiment.get_signal_representation(dataset_info['exp_representation'], channels=['stress'])
            signal = experiment.get_signal_representation(dataset_info['exp_representation'], channels=dataset_info["cpl_model_channels"])
            if takes_max_length is not None:
                nn_input = nn_input[..., :takes_max_length]
                stress = stress[..., :takes_max_length]
                signal = signal[..., :takes_max_length]
            return nn_input, signal, stress

        return get_prepared_experiment


class LossFuncTask(ModuleTask):
    class Meta:
        data_class = InMemoryData
        input_tasks = [UnscaleThetaTorchTask, DatasetInfoTask, ModelFactoryTask]
        parameters = [
            Parameter("loss_func", default='MSELoss'),
            Parameter("loss_func_kwargs", default=None),
        ]

    def run(self, loss_func: str, loss_func_kwargs: dict, unscale_theta_torch, dataset_info: dict,
            model_factory: CPLModelFactory) -> Callable:
        if loss_func_kwargs is None:
            loss_func_kwargs = {}
        # if loss_func == 'MSELoss':
        #     def loss_func(y_pred, y_true, x_true, signal):
        #         loss_val = F.mse_loss(y_pred, y_true, reduction="mean")
        #         return loss_val, {'Loss/train': loss_val.item()}
        #
        #     return loss_func

        if loss_func == 'MSELoss':
            def loss_func(y_pred, y_true, x_true, signal):

                y_pred_unscaled = unscale_theta_torch(y_pred)

                y_mse_loss = torch.Tensor([0]).to(y_pred_unscaled.device)
                if (y_scale := eval(str(loss_func_kwargs.get('y_scale', 1)))) > 0:
                    if (y_crop := loss_func_kwargs.get('y_crop')) is not None:
                        y_mse_loss = y_scale * F.mse_loss(y_pred[:y_crop], y_true[:y_crop], reduction="mean")
                    else:
                        y_mse_loss = y_scale * F.mse_loss(y_pred, y_true, reduction="mean")

                x_mse_loss = torch.Tensor([0]).to(y_pred_unscaled.device)
                if (x_scale := eval(str(loss_func_kwargs.get('x_scale', 0)))) > 0:
                    if np.random.rand() <= loss_func_kwargs.get('x_chance', 1):
                        num_model = model_factory.make_model(theta=y_pred_unscaled)
                        x_pred = num_model.predict_stress_torch_batch(signal=signal.float().to(y_pred_unscaled.device))
                        x_mse_loss = x_scale * F.mse_loss(
                            x_pred[:, :], x_true[:, 0, :], reduction="mean") / (dataset_info['x_std']**2)

                mixed_mse_loss = y_mse_loss + x_mse_loss
                return mixed_mse_loss, {'Loss/train': mixed_mse_loss.item(), 'Loss/train/stress': x_mse_loss.item(),
                                        'Loss/train/theta': y_mse_loss.item()}

            return loss_func
        else:
            raise ValueError(f'Unknown loss function: {loss_func}')


class TrainModelTask(ModuleTask):
    class Meta:
        input_tasks = [
            DatasetInfoTask,
            TrainDatasetTask,
            ValDatasetTask,
            GetRandomPseudoExperimentTask,
            ModelFactoryTask,
            LossFuncTask,
            RealExperimentTask,
            UnscaleThetaTorchTask,
        ]
        parameters = [
            Parameter("model"),
            Parameter("persistent_training_params"),
            Parameter("is_trainable", default=True, dont_persist_default_value=True),
            Parameter("do_compile", default=False, ignore_persistence=True),
            Parameter("other_training_params", ignore_persistence=True),
            Parameter("device", default="cuda", ignore_persistence=True),
            Parameter("eval_on_experiments", ignore_persistence=True),
            Parameter('run_name', ignore_persistence=True),
        ]

    def run(self, dataset_info: dict, train_dataset: Dataset, val_dataset: Dataset,
            get_random_pseudo_experiment: Callable, model_factory: CPLModelFactory, loss_func: callable,
            model, do_compile: bool, is_trainable: bool, persistent_training_params: dict, other_training_params: dict,
            device: str, real_experiment: callable, eval_on_experiments: list, run_name: str,
            unscale_theta_torch) -> DirData:

        measured_experiments = {exp_name: real_experiment(exp_name) for exp_name in eval_on_experiments}

        data: DirData = self.get_data_object()
        if not is_trainable:
            return data

        print(f"\n###\n\nTraining model {run_name} with {repr(model)}\n\n###\n")
        torch.set_float32_matmul_precision('high')
        labels = dataset_info['labels']
        x_metrics = get_metrics(other_training_params.get('metrics', {}).get('x_metrics'))
        x_metrics_items = other_training_params.get('metrics', {}).get('x_metrics_items', np.inf)
        y_metrics = get_metrics(other_training_params.get('metrics', {}).get('y_metrics'))

        torch.manual_seed(1)
        np.random.seed(1)

        runs_dir = RUNS_DIR / run_name
        runs_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(runs_dir)
        reporter = Reporter()

        model = model.to(device)
        if do_compile:
            model = torch.compile(model)

        opt = getattr(torch.optim, persistent_training_params['optim'])(model.parameters(), **persistent_training_params.get('optim_kwargs', {}))

        epochs = persistent_training_params['epochs']
        train_ldr = DataLoader(train_dataset, **persistent_training_params.get('dataloader_kwargs', {}))
        scheduler = eval(persistent_training_params.get('scheduler', 'None'))
        lr_per_epoch_setup = persistent_training_params.get('lr_per_epoch_setup', None)
        if lr_per_epoch_setup is not None:
            assert 0 in lr_per_epoch_setup, 'lr_per_epoch_setup must contain 0 epoch'
        last_lr = None
        nan_grads = 0

        try:
            total_batch_id = 0
            for epoch_id in range(epochs):
                if lr_per_epoch_setup is not None:
                    if epoch_id in lr_per_epoch_setup:
                        for g in opt.param_groups:
                            g['lr'] = lr_per_epoch_setup[epoch_id]
                        last_lr = lr_per_epoch_setup[epoch_id]
                    else:
                        for g in opt.param_groups:
                            g['lr'] = last_lr

                    scheduler = eval(persistent_training_params.get('scheduler', 'None'))

                batch_iterator = tqdm(train_ldr, ncols=90, position=0)
                for batch_id, (batch_x, batch_y, *signal) in enumerate(batch_iterator):
                    if nan_grads > 10:
                        print('Too many NaN gradients. Stopping training.')
                        reporter.save(data.dir / 'reporter.pkl')
                        torch.save(model.state_dict(), data.dir / f'weights.pt')
                        return data
                    x_true = batch_x.to(device)
                    y_true = batch_y.to(device)
                    y_pred = model(x_true)
                    # y_pred.retain_grad()
                    signal = x_true[:, 1:2, :] if len(signal) == 0 else signal[0]

                    loss_val, loss_stat = loss_func(y_pred, y_true, x_true, signal)

                    opt.zero_grad()
                    loss_val.backward()  # compute gradients
                    for loss_name, loss_value in loss_stat.items():
                        reporter.add_mean(loss_name, loss_value)

                    with torch.no_grad():
                        for exp_name, (_nn_input, _signal, _stress) in measured_experiments.items():
                            _theta_hat = model(torch.unsqueeze(torch.from_numpy(_nn_input), 0).float().to(device))
                            _stress_hat = get_random_pseudo_experiment(scaled_theta=_theta_hat[0].cpu().numpy(), signal_correct_representation=_signal)
                            reporter.add_mean(f'exp/{exp_name}', np.mean((_stress_hat - _stress) ** 2))
                            if (total_batch_id + 1) % 10 == 0:
                                unscaled_theta = unscale_theta_torch(_theta_hat).cpu().numpy()
                                _, new_values, old_values = simplex(unscaled_theta=unscaled_theta, true_stress=np.expand_dims(_stress, 0), signal=np.expand_dims(_signal, 0),
                                                                    model_factory=model_factory, verbose=False)
                                reporter.add_mean(f'exp/{exp_name}o', old_values[0])
                                reporter.add_mean(f'exp/{exp_name}n', new_values[0])

                        for metric_name, metric_func in y_metrics.items():
                            reporter.add_mean(metric_name, metric_func(y_pred=y_pred, y_true=y_true))

                        if labels is not None:
                            for param_value, param_label in zip(
                                    torch.mean((y_pred - y_true) ** 2, dim=0).cpu().numpy().tolist(), labels):
                                reporter.add_mean(f'{param_label}/train', param_value)

                        # comparing x_pred and x_true
                        if len(x_metrics) > 0:
                            for item_id, (x_i_true, y_i_pred_npy, signal_i) in enumerate(zip(x_true.detach(),
                                          y_pred.detach().cpu().numpy(), signal.cpu().numpy())):
                                x_i_pred = torch.from_numpy(
                                    get_random_pseudo_experiment(scaled_theta=y_i_pred_npy, signal_correct_representation=signal_i)).to(device)

                                for metric_name, metric_func in x_metrics.items():
                                    # if item_id == 0:
                                    #     print(f'\nx_i_true {x_i_true.shape}: {x_i_true[0, :50]}')
                                    #     print(f'x_i_pred{x_i_pred.shape}: {x_i_pred[:50]}')
                                    #     print(f'metric: {metric_func(x_pred=x_i_pred, x_true=x_i_true[0])}')
                                    reporter.add_mean(metric_name, metric_func(x_pred=x_i_pred, x_true=x_i_true[0]))

                                if item_id > x_metrics_items:
                                    break

                        if (total_batch_id + 1) % other_training_params.get('save_metrics_n', np.inf) == 0:

                            writer.add_scalar('LR', scheduler.get_last_lr()[-1], total_batch_id)

                            for loss_name in loss_stat.keys():
                                writer.add_scalar(loss_name, reporter.get_mean(loss_name), total_batch_id)

                            for exp_name in measured_experiments.keys():
                                writer.add_scalar(f'exp/{exp_name}', reporter.get_mean(f'exp/{exp_name}'), total_batch_id)
                                writer.add_scalar(f'exp/{exp_name}o', reporter.get_mean(f'exp/{exp_name}o'), total_batch_id)
                                writer.add_scalar(f'exp/{exp_name}n', reporter.get_mean(f'exp/{exp_name}n'), total_batch_id)
                            if labels is not None:
                                for param_label in labels:
                                    writer.add_scalar(f'{param_label}/train',
                                                      reporter.get_mean(f'{param_label}/train'), total_batch_id)

                            for metric_name, metric_func in chain(x_metrics.items(), y_metrics.items()):
                                writer.add_scalar(f'{metric_name}/train', reporter.get_mean(metric_name),
                                                  total_batch_id)

                        if (total_batch_id + 1) % other_training_params.get('evaluate_n', np.inf) == 0:
                            for metric_name, metric_val in evaluate_model(
                                        model_factory=model_factory,
                                        dataset_info=dataset_info,
                                        x_metrics=x_metrics,
                                        y_metrics=y_metrics,
                                        model=model,
                                        val_dataset=val_dataset,
                                        device=device,
                                        valid_dataloader_kwargs=other_training_params.get('valid_dataloader_kwargs', {}),
                                        get_random_pseudo_experiment=get_random_pseudo_experiment,
                                    ).items():
                                writer.add_scalar(metric_name, metric_val, total_batch_id)
                                reporter.add_scalar(metric_name, metric_val, total_batch_id)

                    if (total_batch_id + 1) % other_training_params.get('checkpoint_n', np.inf) == 0:
                        torch.save(model.state_dict(), data.dir / f'weights_{total_batch_id}.pt')
                    if (exec_str := persistent_training_params.get('exec_str')) is not None:
                        exec(exec_str)  # execute some code after gradients are computed
                    if contains_nan_gradients(model):
                        nan_grads += 1
                        print("NaN gradients detected. Skipping step.")
                    else:
                        nan_grads = 0
                        opt.step()  # update weights

                    batch_iterator.set_description(
                        f'E{epoch_id + 1}/{persistent_training_params["epochs"]}, '
                        f'LR: {scheduler.get_last_lr()[-1] if scheduler is not None else last_lr: .5f}, '
                        f'Loss: {loss_val.item(): .5f}')
                    total_batch_id += 1
                    if scheduler is not None:
                        scheduler.step()

            reporter.save(data.dir / 'reporter.pkl')
            torch.save(model.state_dict(), data.dir / f'weights.pt')

        except KeyboardInterrupt:
            raise KeyboardInterrupt
            return data
        return data


class ChooseNet:
    def __init__(self, model, do_compile: bool, device: str, train_model: Path):
        self.model = model
        self.do_compile = do_compile
        self.device = device
        self.train_model = train_model

        self.model = self.model.to(self.device)
        if self.do_compile:
            self.model = torch.compile(self.model)

    @property
    def available_checkpoints(self):
        return sorted([int(str(f).split('_')[-1].split('.')[0]) for f in self.train_model.glob('weights_*.pt')])

    def __call__(self, chosen_checkpoint: int | None):
        if chosen_checkpoint is not None:
            weights_path = self.train_model / f'weights_{chosen_checkpoint}.pt'
        else:
            weights_path = self.train_model / 'weights.pt'

        self.model.load_state_dict(torch.load(str(weights_path)))
        return self.model


class ChooseModelTask(ModuleTask):
    class Meta:
        data_class = InMemoryData
        input_tasks = [
            TrainModelTask,
        ]
        parameters = [
            Parameter("model"),
            Parameter("do_compile", default=False, ignore_persistence=True),
            Parameter("device", default="cuda", ignore_persistence=True),
        ]

    def run(self, train_model: Path, model, do_compile: bool, device: str,
            ) -> ChooseNet:
        return ChooseNet(model, do_compile, device, train_model)


class TrainedModelTask(ModuleTask):
    class Meta:
        data_class = InMemoryData
        input_tasks = [
            TrainModelTask,
        ]
        parameters = [
            Parameter("model"),
            Parameter("is_trainable", default=True, dont_persist_default_value=True),
            Parameter("chosen_checkpoint", default=None),
            Parameter("do_compile", default=True, ignore_persistence=True),
            Parameter("device", default="cuda", ignore_persistence=True),
        ]

    def run(self, train_model: Path, model, do_compile: bool, device: str,
            is_trainable: bool,
            chosen_checkpoint: int | None) -> object:
        if chosen_checkpoint is not None:
            weights_path = train_model / f'weights_{chosen_checkpoint}.pt'
        else:
            weights_path = train_model / 'weights.pt'
        model = model.to(device)
        if do_compile:
            model = torch.compile(model)
        if is_trainable:
            model.load_state_dict(torch.load(str(weights_path)))
        return model


class ValidateCrlbTask(ModuleTask):
    class Meta:
        input_tasks = [RealExperimentTask, GetCrlbTask, TrainedModelTask, UnscaleThetaTorchTask, DatasetInfoTask]
        parameters = [
            Parameter("device", default="cuda", ignore_persistence=True),
            Parameter("crlb_runs", default=1000),
            Parameter("takes_channels"),
        ]

    def run(self, real_experiment, get_crlb, trained_model, device, unscale_params_torch, crlb_runs,
            dataset_info, takes_channels) -> dict:
        out = {}
        trained_model.eval()
        all_unscaled_theta = torch.zeros((crlb_runs, 1 + 2 * dataset_info['dim'] + dataset_info['kappa_dim']))
        for experiment_name, sig in real_experiment.items():
            with torch.no_grad():
                theta_hat = trained_model(torch.unsqueeze(torch.from_numpy(sig), 0).float().to(device))[0]
            epsp = REAL_EXPERIMENT[experiment_name]['epsp']
            unscaled_theta = unscale_params_torch(theta_hat)
            crlb = get_crlb(epsp, unscaled_theta)
            with torch.no_grad():
                for i in trange(crlb_runs):
                    # uniform random increment
                    random_increment = torch.randn_like(theta_hat) * torch.from_numpy(crlb['std']).to(device) / 10000
                    unscaled_theta_mod = torch.clip(unscaled_theta + random_increment, min=1e-9, max=None)
                    sig_hat = predict_stress_torch(theta=unscaled_theta_mod, epsp=torch.tensor(epsp).to(device),
                                             dim=dataset_info['dim'], kappa_dim=dataset_info['kappa_dim'])
                    if takes_epsp:
                        sig_hat = torch.unsqueeze(torch.vstack([sig_hat, torch.tensor([epsp]).to(device)]), 0).float()
                    else:
                        sig_hat = torch.unsqueeze(torch.unsqueeze(sig_hat.float(), 0), 0).to(device)
                    theta_hat_hat = trained_model(sig_hat)[0]
                    all_unscaled_theta[i] = (unscale_params_torch(theta_hat_hat) - random_increment).cpu()

            out[experiment_name] = {
                'theta_mean': torch.mean(all_unscaled_theta, dim=0).numpy().tolist(),
                'theta_std': torch.std(all_unscaled_theta, dim=0).numpy().tolist(),
                'theta_relative_std': (
                        torch.std(all_unscaled_theta, dim=0) / unscaled_theta.detach().cpu()).numpy().tolist(),
                'crlb': {key: val.tolist() for key, val in crlb.items()},
            }

        return out


class ModelMetricsTask(ModuleTask):
    class Meta:
        input_tasks = [
            TrainedModelTask,
            DatasetInfoTask,
            TestDatasetTask,
            GetRandomPseudoExperimentTask,
            ScaleThetaTorchTask,
            UnscaleThetaTorchTask,
            RealExperimentTask,
            ModelFactoryTask,
        ]
        parameters = [
            Parameter("other_training_params", ignore_persistence=True),
            Parameter("device", default="cuda", ignore_persistence=True),
            Parameter("prediction_is_scaled", default=True, dont_persist_default_value=True),
            Parameter("evaluate_limit", default=1024),
            Parameter("eval_on_experiments"),
        ]

    def run(self, trained_model, dataset_info: dict, test_dataset: Dataset, get_random_pseudo_experiment: Callable, real_experiment,
            evaluate_limit: int, device: str, scale_theta_torch, unscale_theta_torch, prediction_is_scaled, other_training_params,
            model_factory: CPLModelFactory, eval_on_experiments) -> dict:
        measured_experiments = {exp_name: real_experiment(exp_name) for exp_name in eval_on_experiments}

        x_metrics = get_metrics(other_training_params.get('metrics', {}).get('x_metrics'))
        y_metrics = get_metrics(other_training_params.get('metrics', {}).get('y_metrics'))

        torch.manual_seed(1)
        np.random.seed(1)

        trained_model.eval()

        dataset = Subset(test_dataset, indices=range(evaluate_limit)) if evaluate_limit > 0 else test_dataset
        metric_results = {}
        with torch.no_grad():
            for exp_name, (_nn_input, _signal, _stress) in measured_experiments.items():
                _theta_hat = trained_model(torch.unsqueeze(torch.from_numpy(_nn_input), 0).float().to(device))
                _stress_hat = get_random_pseudo_experiment(scaled_theta=_theta_hat[0].cpu().numpy(),
                                                           signal_correct_representation=_signal)
                unscaled_theta = unscale_theta_torch(_theta_hat).cpu().numpy()
                _, new_values, old_values = simplex(unscaled_theta=unscaled_theta,
                                                    true_stress=np.expand_dims(_stress, 0),
                                                    signal=np.expand_dims(_signal, 0),
                                                    model_factory=model_factory, verbose=False)
                metric_results[f'exp/{exp_name}o'] = old_values[0]
                metric_results[f'exp/{exp_name}n'] = new_values[0]
        metric_results |= {
                key: {key: float(val) for key, val in val_dict.items()} for key, val_dict in
                evaluate_model(
                    valid_dataloader_kwargs=other_training_params.get('valid_dataloader_kwargs', {}),
                    model_factory=model_factory,
                    dataset_info=dataset_info,
                    x_metrics=x_metrics,
                    y_metrics=y_metrics,
                    model=trained_model,
                    val_dataset=dataset,
                    device=device,
                    get_random_pseudo_experiment=get_random_pseudo_experiment,
                    eval_type='both',
                    use_tqdm=True,
                    tqdm_message=f'Raw + Simplex',
                    unscale_theta_torch=unscale_theta_torch,
                    scale_theta_torch=scale_theta_torch,
                    prediction_is_scaled=prediction_is_scaled,
                ).items()}
        print(f'metrics: {metric_results}')
        return metric_results


class SingleScoreTask(ModuleTask):
    class Meta:
        input_tasks = [
            ModelMetricsTask,
        ]
        parameters = [
            Parameter("single_score_definition"),
        ]

    def run(self, model_metrics: dict, single_score_definition: str) -> float:
        mm = model_metrics
        return eval(single_score_definition)
