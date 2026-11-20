"""Mô hình GRU: dùng khối chung RecurrentForecaster (src/models/recurrent.py) với cell = 'gru'."""
from src.models.recurrent import RecurrentForecaster

CELL = 'gru'


def build(n_feat, hidden, layers, dropout, horizon, n_fut=0):
    return RecurrentForecaster(CELL, n_feat, hidden, layers, dropout, horizon, n_fut)
