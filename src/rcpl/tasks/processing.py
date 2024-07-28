import hashlib
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from taskchain import Parameter, ModuleTask, DirData, InMemoryData
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from rcpl.config import RUNS_DIR, MEASURED_EXP_DIR
from rcpl.estimation_model import metrics
from rcpl.experiment import Experiment
from rcpl.material_model import CPLModelFactory
from rcpl.tasks.dataset import TrainDatasetTask, ValDatasetTask, GetRandomPseudoExperimentTask, DatasetInfoTask, \
    UnscaleThetaTorchTask, GetCrlbTask, TestDatasetTask, ScaleThetaTorchTask, ModelFactoryTask
from rcpl.utils.reporter import Reporter
from rcpl.utils.simplex import simplex


def hash_numpy_array(array: torch.Tensor) -> list[str]:
    np_array = array.cpu().numpy()
    return [hashlib.sha256(np_array[i].tobytes()).hexdigest()[:16] for i in range(len(np_array))]


def evaluate_model(valid_dataloader_kwargs, model_factory: CPLModelFactory, dataset_info, x_metrics, y_metrics, model,
                   val_dataset, device, get_random_pseudo_experiment, eval_type='raw', use_tqdm=False,
                   tqdm_message=None, unscale_theta_torch=None, scale_theta_torch=None, prediction_is_scaled=True,
                   store_limit=0):
    model.eval()
    batch_hashes = []
    true_vals = {f'true_{param}': [] for param in dataset_info['labels']}

    labels = dataset_info['labels']
    train_ldr = DataLoader(val_dataset, **valid_dataloader_kwargs)

    reporters = [Reporter(store_limit=store_limit), Reporter(store_limit=store_limit)]  # raw, simplex
    mse_loss_func = nn.MSELoss()
    if use_tqdm:
        train_ldr = tqdm(train_ldr, ncols=120, desc=tqdm_message)
    for batch_id, (batch_x, batch_y, *signal) in enumerate(train_ldr):
        batch_hashes.extend(hash_numpy_array(batch_x))
        x_true = batch_x.to(device)
        y_true = batch_y.to(device)
        y_true_unscaled = unscale_theta_torch(y_true.clone())
        for y_true_unscaled_i in y_true_unscaled:
            for param_value, param_name in zip(y_true_unscaled_i.cpu().numpy(), labels):
                true_vals[f'true_{param_name}'].append(param_value)

        with torch.no_grad():
            # comparing x_pred and x_true
            signal = x_true[:, 1:, :] if len(signal) == 0 else signal[0]  # because *signal gives a list of one element
            prediction_start = datetime.now()
            y_pred_raw = model(x_true)
            prediction_end = datetime.now()
            for reporter_id, use_simplex in enumerate([False, True]):
                if use_simplex and eval_type == 'raw':
                    continue
                if not use_simplex and eval_type == 'simplex':
                    continue

                if prediction_is_scaled:
                    unscaled_y_pred = unscale_theta_torch(y_pred_raw.clone())
                else:
                    unscaled_y_pred = y_pred_raw.clone()

                if use_simplex:
                    simplex_start = datetime.now()
                    unscaled_y_pred, new_values, old_values, num_iters, num_funcalls = simplex(
                        unscaled_theta=unscaled_y_pred.cpu().numpy(),
                        true_stress=x_true[:, 0, :].cpu().numpy(),
                        signal=signal.cpu().numpy(),
                        model_factory=model_factory,
                        verbose=False,
                    )
                    simplex_end = datetime.now()
                    unscaled_y_pred = torch.from_numpy(unscaled_y_pred).to(device)
                    reporters[reporter_id].add_data('old_x_l2/val', np.mean(old_values))
                    reporters[reporter_id].add_data('new_x_l2/val', np.mean(new_values))
                    reporters[reporter_id].add_data('simp_time/val', (simplex_end - simplex_start).total_seconds() / len(new_values))
                    for num_iter, num_funcall in zip(num_iters, num_funcalls):
                        reporters[reporter_id].add_data('num_iter/val', num_iter)
                        reporters[reporter_id].add_data('num_funcall/val', num_funcall)
                    y_pred_scaled = scale_theta_torch(unscaled_y_pred)

                else:
                    if prediction_is_scaled:
                        y_pred_scaled = y_pred_raw.clone()
                    else:
                        y_pred_scaled = scale_theta_torch(y_pred_raw.clone())

                reporters[reporter_id].add_data('loss/val/_all', mse_loss_func(y_pred_scaled, y_true).item())
                reporters[reporter_id].add_data('pred_time/val', (prediction_end - prediction_start).total_seconds() / len(y_true))
                for metric_name, metric_func in y_metrics.items():
                    reporters[reporter_id].add_data(f'{metric_name}/val', metric_func(y_pred=y_pred_scaled, y_true=y_true))

                if labels is not None:
                    for param_value, param_name in zip(
                            torch.mean((y_pred_scaled - y_true) ** 2, dim=0).cpu().numpy(), labels):
                        reporters[reporter_id].add_data(f'loss/val/{param_name}', param_value)
                    for pred_param in unscaled_y_pred:
                        for param_value, param_name in zip(pred_param.cpu().numpy(), labels):
                            reporters[reporter_id].add_data(f'value/val/{param_name}', param_value)

                # comparing x_pred and x_true
                if len(x_metrics) > 0:
                    for item_id, (x_i_true, y_i_pred_npy, signal_i) in enumerate(zip(x_true.detach(),
                                                                                     y_pred_scaled.detach().cpu().numpy(),
                                                                                     signal.cpu().numpy())):
                        x_i_pred = torch.from_numpy(
                            get_random_pseudo_experiment(theta=y_i_pred_npy, signal_correct_representation=signal_i,
                                                         is_scaled=True)).to(device)

                        for metric_name, metric_func in x_metrics.items():
                            reporters[reporter_id].add_data(f'{metric_name}/val', metric_func(x_pred=x_i_pred, x_true=x_i_true[:1]))

    model.train()
    result = {
        'meta': {
            'batch_hashes': batch_hashes,
            'batch_hash': hashlib.sha256(''.join(batch_hashes).encode()).hexdigest()[:16],
        } | true_vals}
    assert eval_type in ['raw', 'simplex', 'both']
    if eval_type == 'raw' or eval_type == 'both':
        result['raw_data'] = reporters[0].get_stored()
        result['raw'] = reporters[0].get_mean()
    if eval_type == 'simplex' or eval_type == 'both':
        result['simplex_data'] = reporters[1].get_stored()
        result['simplex'] = reporters[1].get_mean()
    return result


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
        def get_prepared_experiment(experiment_path: Path | str, max_len=None):
            if isinstance(experiment_path, str):
                if not experiment_path.endswith('.json'):
                    experiment_path += '.json'
                experiment_path = MEASURED_EXP_DIR / experiment_path
            experiment = Experiment(json_path=experiment_path)
            nn_input = experiment.get_signal_representation(dataset_info['exp_representation'], channels=takes_channels)
            stress = experiment.get_signal_representation(dataset_info['exp_representation'], channels=['stress'])
            signal = experiment.get_signal_representation(dataset_info['exp_representation'],
                                                          channels=dataset_info["cpl_model_channels"])

            max_len = max_len if max_len is not None else takes_max_length
            max_len = max(max_len, takes_max_length) if takes_max_length is not None else max_len
            if takes_max_length is not None:
                nn_input = nn_input[..., :takes_max_length]
                stress = stress[..., :max_len]
                signal = signal[..., :max_len]
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
                            x_pred[:, :], x_true[:, 0, :], reduction="mean") / (dataset_info['x_std'] ** 2)

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

        opt = getattr(torch.optim, persistent_training_params['optim'])(model.parameters(),
                                                                        **persistent_training_params.get('optim_kwargs',
                                                                                                         {}))

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
                        reporter.add_data(loss_name, loss_value)

                    with torch.no_grad():
                        for exp_name, (_nn_input, _signal, _stress) in measured_experiments.items():
                            _theta_hat = model(torch.unsqueeze(torch.from_numpy(_nn_input), 0).float().to(device))
                            _stress_hat = get_random_pseudo_experiment(theta=_theta_hat[0].cpu().numpy(),
                                                                       signal_correct_representation=_signal,
                                                                       is_scaled=True)
                            reporter.add_data(f'exp/{exp_name}', np.mean((_stress_hat - _stress) ** 2))
                            if (total_batch_id + 1) % 10 == 0:
                                unscaled_theta = unscale_theta_torch(_theta_hat).cpu().numpy()
                                _, new_values, old_values, _, _ = simplex(unscaled_theta=unscaled_theta,
                                                                          true_stress=np.expand_dims(_stress, 0),
                                                                          signal=np.expand_dims(_signal, 0),
                                                                          model_factory=model_factory, verbose=False)
                                reporter.add_data(f'exp/{exp_name}o', old_values[0])
                                reporter.add_data(f'exp/{exp_name}n', new_values[0])

                        for metric_name, metric_func in y_metrics.items():
                            reporter.add_data(metric_name, metric_func(y_pred=y_pred, y_true=y_true))

                        if labels is not None:
                            for param_value, param_label in zip(
                                    torch.mean((y_pred - y_true) ** 2, dim=0).cpu().numpy().tolist(), labels):
                                reporter.add_data(f'{param_label}/train', param_value)

                        # comparing x_pred and x_true
                        if len(x_metrics) > 0:
                            for item_id, (x_i_true, y_i_pred_npy, signal_i) in enumerate(zip(x_true.detach(),
                                                                                             y_pred.detach().cpu().numpy(),
                                                                                             signal.cpu().numpy())):
                                x_i_pred = torch.from_numpy(
                                    get_random_pseudo_experiment(theta=y_i_pred_npy,
                                                                 signal_correct_representation=signal_i,
                                                                 is_scaled=True)).to(device)

                                for metric_name, metric_func in x_metrics.items():
                                    # if item_id == 0:
                                    #     print(f'\nx_i_true {x_i_true.shape}: {x_i_true[0, :50]}')
                                    #     print(f'x_i_pred{x_i_pred.shape}: {x_i_pred[:50]}')
                                    #     print(f'metric: {metric_func(x_pred=x_i_pred, x_true=x_i_true[0])}')
                                    reporter.add_data(metric_name, metric_func(x_pred=x_i_pred, x_true=x_i_true[0]))

                                if item_id > x_metrics_items:
                                    break

                        if (total_batch_id + 1) % other_training_params.get('save_metrics_n', np.inf) == 0:

                            writer.add_scalar('LR', scheduler.get_last_lr()[-1], total_batch_id)

                            for loss_name in loss_stat.keys():
                                writer.add_scalar(loss_name, reporter.get_mean(loss_name), total_batch_id)

                            for exp_name in measured_experiments.keys():
                                writer.add_scalar(f'exp/{exp_name}', reporter.get_mean(f'exp/{exp_name}'),
                                                  total_batch_id)
                                writer.add_scalar(f'exp/{exp_name}o', reporter.get_mean(f'exp/{exp_name}o'),
                                                  total_batch_id)
                                writer.add_scalar(f'exp/{exp_name}n', reporter.get_mean(f'exp/{exp_name}n'),
                                                  total_batch_id)
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
                                    eval_type='raw',
                                    unscale_theta_torch=unscale_theta_torch,
                            )['raw'].items():
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
                        # print(f'LR: {scheduler.get_last_lr()[-1]}')

            reporter.save(data.dir / 'reporter.pkl')
            torch.save(model.state_dict(), data.dir / f'weights.pt')

        except KeyboardInterrupt:
            raise NotImplementedError
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
            TrainModelTask, ModelFactoryTask,
        ]
        parameters = [
            Parameter("model"),
            Parameter("is_trainable", default=True, dont_persist_default_value=True),
            Parameter("chosen_checkpoint", default=None),
            Parameter("do_compile", default=True, ignore_persistence=True),
            Parameter("device", default="cuda", ignore_persistence=True),
        ]

    def run(self, train_model: Path, model, do_compile: bool, device: str,
            is_trainable: bool, model_factory: CPLModelFactory,
            chosen_checkpoint: int | None) -> object:
        if chosen_checkpoint is not None:
            weights_path = train_model / f'weights_{chosen_checkpoint}.pt'
        else:
            weights_path = train_model / 'weights.pt'
        model = model.to(device)
        if do_compile:
            model = torch.compile(model)
        if is_trainable:
            state_dict = torch.load(str(weights_path))
            try:
                model.load_state_dict(state_dict)
            except:
                # Create a new state dictionary with the correct keys
                new_state_dict = {}
                for key, value in state_dict.items():
                    new_key = key.replace('_orig_mod.', '')
                    new_state_dict[new_key] = value
                model.load_state_dict(new_state_dict)
        else:
            model.model_factory = model_factory
        return model


class ValidateCrlbTask(ModuleTask):
    class Meta:
        data_class = InMemoryData
        input_tasks = [RealExperimentTask, GetCrlbTask, TrainedModelTask, UnscaleThetaTorchTask, DatasetInfoTask,
                       ModelFactoryTask]
        parameters = [
            Parameter("device", default="cuda", ignore_persistence=True),
            Parameter("crlb_runs", default=1000),
            Parameter("takes_channels"),
        ]

    def run(self, real_experiment, get_crlb, trained_model, device, unscale_theta_torch, crlb_runs,
            dataset_info, takes_channels, model_factory: CPLModelFactory) -> Callable:
        def get_crlb_(exp_path, crop_signal, unscaled_theta=None, fraction=0.001):
            trained_model.eval()
            nn_input, signal, stress = real_experiment(exp_path)
            signal = torch.from_numpy(signal)

            if unscaled_theta is None:
                with torch.no_grad():
                    theta_hat = trained_model(torch.unsqueeze(torch.from_numpy(nn_input), 0).float().to(device))

                unscaled_theta = unscale_theta_torch(theta_hat).detach().cpu()[0]
            if isinstance(unscaled_theta, np.ndarray):
                unscaled_theta = torch.from_numpy(unscaled_theta)

            num_model = model_factory.make_model(unscaled_theta.double())
            stress_pred = num_model.predict_stress_torch(signal)

            crlb = get_crlb(signal, unscaled_theta)

            all_unscaled_theta = torch.zeros((crlb_runs, len(unscaled_theta)))
            with torch.no_grad():
                for i in trange(crlb_runs):
                    # # uniform random increment
                    # random_increment = torch.randn_like(unscaled_theta, device='cpu') * torch.from_numpy(crlb['std']).cpu() * fraction
                    # unscaled_theta_mod = torch.clip(unscaled_theta + random_increment, min=1e-9, max=None)
                    # num_model_mod = model_factory.make_model(unscaled_theta_mod)
                    # stress_pred_mod = num_model_mod.predict_stress_torch(signal)
                    # experiment = Experiment(
                    #     signal=torch.vstack([signal, stress_pred_mod]),
                    #     channel_labels=dataset_info['cpl_model_channels'] + ['stress'],
                    #     representation=dataset_info['exp_representation'],
                    #     crop_signal=crop_signal,
                    # )
                    # theta_hat_hat = trained_model(experiment.get_signal_representation(dataset_info['exp_representation'], channels=takes_channels).float().unsqueeze(0).to(device))[0]
                    # all_unscaled_theta[i] = unscale_theta_torch(theta_hat_hat).cpu() - random_increment

                    # uniform random increment
                    random_increment = torch.randn_like(stress_pred, device='cpu') * fraction
                    stress_pred_mod = stress_pred + random_increment
                    experiment = Experiment(
                        signal=torch.vstack([signal, stress_pred_mod]),
                        channel_labels=dataset_info['cpl_model_channels'] + ['stress'],
                        representation=dataset_info['exp_representation'],
                        crop_signal=crop_signal,
                    )
                    theta_hat_hat = trained_model(
                        experiment.get_signal_representation(dataset_info['exp_representation'],
                                                             channels=takes_channels).float().unsqueeze(0).to(device))[
                        0]
                    all_unscaled_theta[i] = unscale_theta_torch(theta_hat_hat).cpu()

            return {
                'theta_mean': torch.mean(all_unscaled_theta, dim=0).numpy().tolist(),
                'theta_std': torch.std(all_unscaled_theta, dim=0).numpy().tolist(),
                'theta_relative_std': (
                        torch.std(all_unscaled_theta, dim=0) / unscaled_theta).numpy().tolist(),
                'crlb': {key: val.tolist() for key, val in crlb.items()},
                'unscaled_theta': unscaled_theta.numpy().tolist(),
                'stress_pred': stress_pred.tolist(),
                'stress_true': stress[0].tolist(),
                'signal0': signal[0].tolist(),
            }

        return get_crlb_


def floatify(x: float | list | dict):
    if isinstance(x, list):
        return [floatify(xi) for xi in x]
    elif isinstance(x, dict):
        return {key: floatify(val) for key, val in x.items()}
    else:
        if isinstance(x, str):
            return x
        return float(x)


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

    def run(self, trained_model, dataset_info: dict, test_dataset: Dataset, get_random_pseudo_experiment: Callable,
            real_experiment,
            evaluate_limit: int, device: str, scale_theta_torch, unscale_theta_torch, prediction_is_scaled,
            other_training_params,
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
                if prediction_is_scaled:
                    unscaled_theta = unscale_theta_torch(_theta_hat).cpu().numpy()
                else:
                    unscaled_theta = _theta_hat.cpu().numpy()

                _stress_hat = get_random_pseudo_experiment(theta=_theta_hat[0].cpu().numpy(),
                                                           signal_correct_representation=_signal,
                                                           is_scaled=prediction_is_scaled)
                _, new_values, old_values, num_iters, num_funcalls = simplex(unscaled_theta=unscaled_theta,
                                                                             true_stress=np.expand_dims(_stress, 0),
                                                                             signal=np.expand_dims(_signal, 0),
                                                                             model_factory=model_factory,
                                                                             verbose=False)
                metric_results[f'exp/{exp_name}o'] = old_values[0]
                metric_results[f'exp/{exp_name}n'] = new_values[0]
                metric_results[f'exp/{exp_name}i'] = num_iters[0]
                metric_results[f'exp/{exp_name}f'] = num_funcalls[0]
        metric_results |= {
            key: {key: floatify(val) for key, val in val_dict.items()} for key, val_dict in
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
                store_limit=20000,
            ).items()}
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
