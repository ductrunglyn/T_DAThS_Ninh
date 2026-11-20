"""Persistence: Q̂(t+h) = Q đến CĐ gần nhất đã biết tại t (giá trị điền nhân quả nếu ngày t thiếu)."""
import numpy as np


class Persistence:
    supports_weights = False
    def fit(self, v, idx, Yz, w=None, eval_set=None):
        return self
    def predict(self, v, idx):
        return np.repeat(v.qlast[idx][:, None], v.Y.shape[1], axis=1)
