"""Ridge-ARX: hồi quy tuyến tính có điều chuẩn L2 trên các độ trễ (X_flat) của Q, mưa và vận hành Hủa Na."""
from sklearn.linear_model import Ridge


class RidgeARX:
    supports_weights = True
    def __init__(self, alpha=1.0, **_):
        self.m = Ridge(alpha=alpha)
    def fit(self, v, idx, Yz, w=None, eval_set=None):
        self.m.fit(v.X_flat[idx], Yz, sample_weight=w)
        return self
    def predict(self, v, idx):
        return v.to_q(self.m.predict(v.X_flat[idx]).reshape(len(idx), -1))
