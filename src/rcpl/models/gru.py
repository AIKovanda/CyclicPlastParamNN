import numpy as np
import torch
from torch import nn

from rcpl.models.inceptiontime import InceptionBlock
from rcpl.models.tools import get_activation


class GRU(nn.Module):
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
        self.rnn = nn.GRU(in_channels, hidden_size, layers, batch_first=True, **rnn_kwargs)

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


class InceptionGRU(nn.Module):
    def __init__(self, in_channels, hidden_size, layers, outputs, kernel_sizes, bias, combine=True, pool_size=1):
        assert hidden_size % 4 == 0
        super(InceptionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.bias = bias
        self.layers = layers
        self.pool_size = pool_size
        self.outputs = outputs
        self.combine = combine
        self.kernel_sizes = kernel_sizes
        n_filters = 32
        self.inc1 = InceptionBlock(
                in_channels=in_channels,
                n_filters=n_filters,
                kernel_sizes=kernel_sizes,
                bottleneck_channels=32,
                use_residual=False,
                use_batch_norm=False,
                activation=get_activation('tanh'),
                bias=bias,
            )
        if self.combine:
            self.inc2 = InceptionBlock(
                    in_channels=n_filters * 4 + in_channels,
                    n_filters=int(hidden_size/4),
                    kernel_sizes=kernel_sizes,
                    bottleneck_channels=32,
                    use_residual=False,
                    use_batch_norm=False,
                    activation=get_activation('tanh'),
                    bias=bias,
                )
        self.rnn = nn.GRU(n_filters * 4 + in_channels, hidden_size, layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*4*self.pool_size, outputs)
        self.max = nn.AdaptiveMaxPool1d(output_size=self.pool_size)
        self.avg = nn.AdaptiveAvgPool1d(output_size=self.pool_size)

    def forward(self, x: torch.Tensor):
        h0 = torch.zeros(self.layers, x.size(0), self.hidden_size).to(x.device)
        x0 = self.inc1(x)
        x = torch.concat([x, x0], dim=1)
        out_rnn, _ = self.rnn(torch.swapaxes(x, 1, 2), h0)
        rnn_out_swapped = torch.swapaxes(out_rnn, 1, 2)
        outs = [self.avg(rnn_out_swapped).view(-1, self.hidden_size*self.pool_size),
                self.max(rnn_out_swapped).view(-1, self.hidden_size*self.pool_size)]
        if self.combine:
            out_inc = self.inc2(x)
            outs.append(self.avg(out_inc).view(-1, self.hidden_size*self.pool_size))
            outs.append(self.max(out_inc).view(-1, self.hidden_size*self.pool_size))
        return self.fc(torch.concat(outs, dim=-1))
