"""Khối dự báo hồi quy dùng chung cho RNN / LSTM / GRU: chuỗi → trạng thái ẩn cuối → (ghép mưa tương lai) → Linear(H)."""
import torch
import torch.nn as nn

CELLS = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}


class RecurrentForecaster(nn.Module):
    def __init__(self, cell, n_feat, hidden, layers, dropout, horizon, n_fut=0):
        super().__init__()
        kw = dict(batch_first=True, dropout=dropout if layers > 1 else 0.0)
        if cell == 'rnn':
            kw['nonlinearity'] = 'tanh'
        self.rnn = CELLS[cell](n_feat, hidden, layers, **kw)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden + n_fut, horizon)

    def forward(self, x, fut=None):
        out, _ = self.rnn(x)
        z = self.drop(out[:, -1])
        if fut is not None:
            z = torch.cat([z, fut], dim=1)
        return self.head(z)
