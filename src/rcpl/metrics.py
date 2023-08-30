import numpy as np
import torch


def x_l(x_pred: torch.Tensor, x_true: torch.Tensor, p=2, **kwargs) -> float:
    return float(torch.norm(x_pred - x_true, p=p))


def y_l(y_pred: torch.Tensor, y_true: torch.Tensor, p=2, **kwargs) -> np.array:
    return torch.norm(y_pred - y_true, p=p, dim=0).cpu().numpy()


def x_collect(x_pred: torch.Tensor, x_true: torch.Tensor, **kwargs) -> np.ndarray:
    return torch.stack([x_pred, x_true], dim=0).cpu().numpy()
