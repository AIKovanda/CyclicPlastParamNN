import torch
from taskchain.parameter import AutoParameterObject
from torch import nn


class FFN(AutoParameterObject, nn.Module):
    def __init__(self, in_channels, inputs, layers: list[int], outputs, batchnorm=True):
        super().__init__()
        self.in_channels = in_channels
        self.inputs = inputs
        self.layers = layers
        self.outputs = outputs
        self.batchnorm = batchnorm

        self.bn_layer = nn.BatchNorm1d(in_channels)
        self.nn_layers = nn.ModuleList()
        for i, layer in enumerate(layers):
            self.nn_layers.append(nn.Linear(inputs * in_channels if i == 0 else layers[i - 1], layer))
            self.nn_layers.append(nn.ReLU())
        self.nn_layers.append(nn.Linear(layers[-1], outputs))

    def forward(self, x: torch.Tensor):
        if self.batchnorm:
            x = self.bn_layer(x)
        x = x.reshape(x.shape[0], -1)
        for layer in self.nn_layers:
            x = layer(x)
        return x
