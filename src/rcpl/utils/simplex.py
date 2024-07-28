from functools import partial

import numpy as np
from scipy.optimize import fmin

from rcpl.material_model import CPLModelFactory


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
        np.ndarray, list, list, list, list]:
    opt_theta = np.zeros_like(unscaled_theta)
    new_values = []
    old_values = []
    num_iters = []
    num_funcalls = []
    for i, (theta_i, true_stress_i, signal_i) in enumerate(zip(unscaled_theta, true_stress, signal)):
        new_theta_i, new_value, now_value, num_iter, num_funcall = simplex_one(theta_i, true_stress_i, signal_i, model_factory=model_factory, verbose=verbose)
        opt_theta[i] = new_theta_i
        # print('New:', new_value)
        # print('Old:', now_value)
        new_values.append(new_value)
        old_values.append(now_value)
        num_iters.append(num_iter)
        num_funcalls.append(num_funcall)
    return opt_theta, new_values, old_values, num_iters, num_funcalls


def simplex_one(unscaled_theta: np.ndarray, true_stress: np.ndarray, signal: np.ndarray, model_factory: CPLModelFactory, verbose=True, **fmin_kwargs):
    if not validate(unscaled_theta, model_factory.simplex_lower_bound, model_factory.simplex_upper_bound):
        more_than_bound = unscaled_theta - model_factory.simplex_upper_bound
        more_than_bound[more_than_bound < 0] = 0
        less_than_bound = model_factory.simplex_lower_bound - unscaled_theta
        less_than_bound[less_than_bound < 0] = 0
        if verbose:
            print(f'BUG - not valid - clipping...\n'
                  f'Above bound: {more_than_bound}\n'
                  f'Below bound: {less_than_bound}')
        unscaled_theta = np.clip(unscaled_theta, model_factory.simplex_lower_bound + 1e-9, model_factory.simplex_upper_bound - 1e-9)
    assert validate(unscaled_theta, model_factory.simplex_lower_bound, model_factory.simplex_upper_bound)
    now_value = to_optimize_one(unscaled_theta, true_stress, signal, model_factory=model_factory)
    # print('Now:', now_value)
    new_theta_i, f_min, num_iter, num_funcall, _ = fmin(
        partial(to_optimize_one, true_stress=true_stress, signal_=signal, model_factory=model_factory),
        unscaled_theta, disp=0, full_output=True, **fmin_kwargs)
    assert validate(new_theta_i, model_factory.simplex_lower_bound, model_factory.simplex_upper_bound)
    if now_value < f_min:
        new_theta_i = unscaled_theta
        print('BUG - new is worse')
    return new_theta_i, f_min, now_value, num_iter, num_funcall
