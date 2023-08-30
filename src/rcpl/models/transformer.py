import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):  # documentation code
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 1000):  #
        super(PositionalEncoding, self).__init__()  # old syntax
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # like 10x4
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # allows state-save

    def forward(self, x):
        x = x + self.pe[..., :x.size(1), :]
        return self.dropout(x)


class TransformerNet(nn.Module):
    name = 'transformer'

    def __init__(self, d_model=32, in_dim=1, enc_dim=20, str_len=991, nhead=2, num_layers=6, outs=11, embedding=None):
        # vocab_size = 12, embed_dim = d_model = 4, seq_len = 9/10
        super().__init__()  # classic syntax
        if embedding is not None:
            self.embedding = nn.Sequential(nn.Linear(in_dim + enc_dim, embedding), nn.ReLU(),
                                           nn.Linear(embedding, d_model))
        else:
            self.embedding = None
            assert d_model == in_dim + enc_dim, f'{d_model=}, {in_dim=}, {enc_dim=}'

        self.d_model = d_model
        self.src_tok_embedding = nn.Embedding(str_len, enc_dim)
        self.pos_enc = PositionalEncoding(d_model)  # positional
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.1, batch_first=True,
                                                        norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model * str_len, outs)  # embed_dim to vocab_size

        self.batchnorm = nn.BatchNorm1d(str_len)

    def forward(self, s):  # src: (batch x str_len)
        s = torch.moveaxis(s, 2, 1)
        s = self.batchnorm(s)
        pos_src = self.src_tok_embedding(torch.arange(0, s.shape[1]).unsqueeze(0).repeat(s.shape[0], 1).to(s.device))
        s = torch.cat([s, pos_src], dim=2)
        if self.embedding is not None:
            s = self.embedding(s)
        # s = self.pos_enc(s)   # (batch x str_len x d_model)
        z = self.transformer_encoder(s)
        z = self.fc(torch.reshape(z, (s.size(0), -1)))
        return z
