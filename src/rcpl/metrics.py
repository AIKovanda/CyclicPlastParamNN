import numpy as np
import torch


def x_l2(x_pred: torch.Tensor, x_true: torch.Tensor, **kwargs) -> float:
    return float(torch.norm(x_pred - x_true, p=2))


def y_l2(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> np.array:
    return torch.norm(y_pred - y_true, p=2, dim=0).cpu().numpy()


def x_collect(x_pred: torch.Tensor, x_true: torch.Tensor, **kwargs) -> np.ndarray:
    return torch.stack([x_pred, x_true], dim=0).cpu().numpy()
