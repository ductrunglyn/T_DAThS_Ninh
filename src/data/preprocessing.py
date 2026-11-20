"""Biến đổi (không phụ thuộc fold) và chuẩn hoá z-score (tham số học từ train của từng fold)."""
import numpy as np
import pandas as pd

from src.data.feature_engineering import FEATURE_META


def transform(features):
    out = pd.DataFrame(index=features.index)
    for c in features.columns:
        t = FEATURE_META[c][1]
        x = features[c]
        if t == 'log1p':
            out[c] = np.log1p(x)
        elif t == 'slog':
            out[c] = np.sign(x) * np.log1p(np.abs(x))
        else:
            out[c] = x
    return out


class Scaler:
    """z = (x − μ_train) / σ_train; biến nhị phân giữ 0/1."""
    def __init__(self, T, train_days):
        self.mu = T.loc[train_days].mean()
        self.sd = T.loc[train_days].std()
        for c in T.columns:
            if FEATURE_META[c][1] == 'binary':
                self.mu[c], self.sd[c] = 0.0, 1.0
        self.sd = self.sd.replace(0, 1.0)

    def __call__(self, T):
        return (T - self.mu) / self.sd
