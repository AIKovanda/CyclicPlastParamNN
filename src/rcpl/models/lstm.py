import numpy as np
import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, in_channels, hidden_size, layers, outputs, rnn_kwargs=None, batchnorm=True, first_last=False,
                 fc_layers: list[int] = None, segments_num=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.batchnorm = batchnorm
        self.segments_num = segments_num
        self.first_last = first_last
        if rnn_kwargs is None:
            rnn_kwargs = {}
        self.rnn = nn.LSTM(in_channels, hidden_size, layers, batch_first=True, **rnn_kwargs)

        self.bn_layer = nn.BatchNorm1d(in_channels)
        # -> x needs to be: (batch_size, seq, input_size)

        # or:
        # self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        fc_sizes = [hidden_size * (2 if rnn_kwargs.get('bidirectional', False) else 1) * (segments_num+1 if segments_num is not None else (2 if first_last else 1))]
        if fc_layers is not None:
            fc_sizes.extend(fc_layers)
        fc_sizes.append(outputs)
        self.fcs = nn.ModuleList([nn.Linear(fc_sizes[i], fc_sizes[i + 1]) for i in range(len(fc_sizes) - 1)])

    def forward(self, x: torch.Tensor):
        if self.batchnorm:
            x = self.bn_layer(x)
        # Set initial hidden states (and cell states for LSTM)
        x = torch.swapaxes(x, 1, 2)
        # h0 = torch.zeros(self.layers, x.size(0), self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.layers, x.size(0), self.hidden_size).to(x.device)

        # x: (n, 28, 28), h0: (2, n, 128)

        # Forward propagate RNN
        out, _ = self.rnn(x)
        # or:
        # out, _ = self.lstm(x, (h0,c0))

        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 28, 128)

        # Decode the hidden state of the last time step
        if self.segments_num is not None:
            taken_indices = np.linspace(0, out.shape[1] - 1, self.segments_num + 1, dtype=int)
            out = torch.cat([out[:, i, :] for i in taken_indices], dim=1)
        else:
            if self.first_last:
                out = torch.cat((out[:, 0, :], out[:, -1, :]), dim=1)
            else:
                out = out[:, -1, :]
        # out: (n, 128)
        for i, fc in enumerate(self.fcs):
            out = fc(out)
            # relu everywhere expect last layer
            if i < len(self.fcs) - 1:
                out = torch.relu(out)
        return out
