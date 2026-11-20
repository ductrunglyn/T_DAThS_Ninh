"""Extra Trees: rừng cây cực ngẫu nhiên trên X_flat (đa đầu ra khi H > 1)."""
from sklearn.ensemble import ExtraTreesRegressor


class ExtraTrees:
    supports_weights = True
    def __init__(self, n_estimators=300, min_samples_leaf=1, max_features=1.0, seed=0, **_):
        self.m = ExtraTreesRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                     max_features=max_features, n_jobs=-1, random_state=seed)
    def fit(self, v, idx, Yz, w=None, eval_set=None):
        self.m.fit(v.X_flat[idx], Yz if Yz.shape[1] > 1 else Yz.ravel(), sample_weight=w)
        return self
    def predict(self, v, idx):
        return v.to_q(self.m.predict(v.X_flat[idx]).reshape(len(idx), -1))
