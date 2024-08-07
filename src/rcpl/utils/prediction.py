import math
import timeit

import torch
from taskchain.task import Config

from rcpl import config
from rcpl.optimization.paper_optim import get_chain
from rcpl.utils.simplex import simplex_one


class ChainPredictor:
    def __init__(self, config_path=None, config_dict=None, predicts_scaled_theta=False, device='cuda'):
        if config_path is not None:
            config_path_ttopt = config_path
            conf_ttopt = Config(
                config.TASKS_DIR,  # where Taskchain data should be stored
                config_path_ttopt,
                context={'device': device},
                global_vars=config,  # set global variables
            )
            self.chain = conf_ttopt.chain()
            self.chain.set_log_level('CRITICAL')
        if config_dict is not None:
            self.chain = get_chain(**config_dict)

        # assert self._chain.train_model.has_data
        self._predicts_scaled_theta = predicts_scaled_theta
        self.device = device

    @property
    def model_metrics(self):
        return self.chain.model_metrics.value

    def get_experiment_data(self, _exp_path, pred_len=None):
        epsp, stress = self.chain.real_experiment.value(_exp_path, max_len=pred_len)[1:]
        return epsp[0], stress[0]

    def predict_from_json(self, _exp_path, pred_len=None):
        model_input, signal, _ = self.chain.real_experiment.value(_exp_path, max_len=pred_len)
        return self.predict_single_data(model_input, signal)

    def predict_single_data(self, model_input, signal):
        net = self.chain.trained_model.value
        torch_exp = torch.unsqueeze(torch.from_numpy(model_input), 0).float().to(self.device)
        net.eval()
        with torch.no_grad():
            theta_hat = net(torch_exp)
        if self._predicts_scaled_theta:
            unscaled_theta_prediction = self.chain.unscale_theta_torch.value(theta_hat).cpu().numpy()[0]
        else:
            unscaled_theta_prediction = theta_hat.cpu().numpy()[0]

        stress_pred = self.predict_single_stress(signal, unscaled_theta_prediction)
        return unscaled_theta_prediction, stress_pred

    def predict_stress(self, _exp_path, theta=None):
        if theta is None:
            theta, _ = self.predict_from_json(_exp_path)
        model_input, signal, stress = self.chain.real_experiment.value(_exp_path)
        num_model = self.chain.model_factory.value.make_model(torch.from_numpy(theta).double().cpu())
        return signal, stress, num_model.predict_stress_torch(torch.from_numpy(signal))

    def predict_from_json_simplex(self, _exp_path, verbose=True, pred_len=None, simplex_len=None, **fmin_kwargs):
        return self.run_simplex_on_exp(_exp_path, self.predict_from_json(_exp_path)[0], verbose=verbose, pred_len=pred_len, simplex_len=simplex_len, **fmin_kwargs)

    def run_simplex_on_exp(self, _exp_path, unscaled_theta_prediction, pred_len=None, simplex_len=None, **fmin_kwargs):
        model_input, simplex_signal, simplex_stress = self.chain.real_experiment.value(_exp_path, max_len=simplex_len)
        unscaled_theta_opt, stress_pred_opt, (n_score, o_score) = self.run_simplex(
            simplex_signal, simplex_stress, unscaled_theta_prediction, **fmin_kwargs)

        _, signal, stress = self.chain.real_experiment.value(_exp_path, max_len=pred_len)
        num_model = self.chain.model_factory.value.make_model(torch.from_numpy(unscaled_theta_opt).double().cpu())
        return unscaled_theta_opt, num_model.predict_stress_torch(torch.from_numpy(signal)), (n_score, o_score)

    def run_simplex(self, signal, stress, unscaled_theta_prediction, verbose=False, **fmin_kwargs):
        unscaled_theta_opt, n_score, o_score, _, _ = simplex_one(unscaled_theta_prediction, stress, signal,
                                                                 self.chain.model_factory.value, **fmin_kwargs)
        if verbose:
            print(f'Origin score: {o_score:.2f}, Nelder-Mead score: {n_score:.2f}')
        stress_pred_opt = self.predict_single_stress(signal, unscaled_theta_opt)
        return unscaled_theta_opt, stress_pred_opt, (n_score, o_score)

    def predict_single_stress(self, signal, unscaled_theta_opt):
        num_model = self.chain.model_factory.value.make_model(torch.from_numpy(unscaled_theta_opt).double().cpu())
        stress_pred_opt = num_model.predict_stress_torch(torch.from_numpy(signal))
        return stress_pred_opt

    def benchmark(self, _exp_path, num_timeit=1):
        model_input, signal, stress = self.chain.real_experiment.value(_exp_path)
        net = self.chain.trained_model.value
        torch_exp = torch.unsqueeze(torch.from_numpy(model_input), 0).float().to(self.device)
        net.eval()
        with torch.no_grad():
            _ = net(torch_exp)
            return timeit.timeit(lambda: net(torch_exp), number=num_timeit) / num_timeit

    def validate_crlb(self, _exp_path, crop_signal=None):
        return self.chain.validate_crlb.value(_exp_path, crop_signal=crop_signal)

    def get_crlb(self):
        return self.chain.get_crlb.value


def format_float(x, valid=4):
    if x >= 10000 or x < 0.001:
        exponent = int(math.log10(abs(x)))
        mantissa = x / (10**exponent)
        return f"{mantissa:.{valid-1}f} $\\times 10^{{{exponent}}}$"
    xx = eval(f"f'{{x:.{valid}g}}'")
    if '.' in xx and len(xx) < valid+1:
        xx += '0' * (valid+1-len(xx))
    return xx
