import torch
from torch import nn


class FC(nn.Module):
    def __init__(self, in_channels, inputs, layers: list[int], outputs, batchnorm=True):
        super().__init__()
        self.batchnorm = batchnorm
        self.layers = nn.ModuleList()
        for i, layer in enumerate(layers):
            self.layers.append(nn.Linear(inputs*in_channels if i == 0 else layers[i - 1], layer))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(layers[-1], outputs))

        self.bn_layer = nn.BatchNorm1d(in_channels)

    def forward(self, x: torch.Tensor):
        print(x.shape)
        if self.batchnorm:
            x = self.bn_layer(x)
        x = x.reshape(x.shape[0], -1)
        for layer in self.layers:
            x = layer(x)
        return x
