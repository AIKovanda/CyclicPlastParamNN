import numpy as np
import torch


def x_l2(x_pred: torch.Tensor, x_true: torch.Tensor, **kwargs) -> float:
    return float(torch.mean((x_pred - x_true)**2))


def x_l2_median(x_pred: torch.Tensor, x_true: torch.Tensor, **kwargs) -> float:
    return float(torch.median(torch.mean((x_pred - x_true)**2, dim=0)))


def y_l2(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> np.array:
    return torch.mean((y_pred - y_true)**2, dim=0).cpu().numpy()


def y_l2_median(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> np.array:
    return torch.median(torch.mean((y_pred - y_true)**2, dim=0), dim=0).cpu().numpy()
