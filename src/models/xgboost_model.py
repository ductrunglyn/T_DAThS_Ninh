"""XGBoost (hist, CPU 16 luồng — với dữ liệu nhỏ CPU nhanh hơn GPU khoảng 5 lần). Ở CV: dừng sớm theo tập val của fold; ở giai đoạn cuối: số cây = trung vị số cây tốt
nhất trên các fold."""
import numpy as np
import xgboost as xgb


class XGBoost:
    supports_weights = True
    def __init__(self, max_depth=6, min_child_weight=1, learning_rate=0.05, max_estimators=2000, early_stopping=50,
                 subsample=0.8, colsample_bytree=0.8, n_estimators=None, seed=0, **_):
        self.fixed_n = n_estimators
        self.kw = dict(max_depth=max_depth, min_child_weight=min_child_weight, learning_rate=learning_rate,
                       subsample=subsample, colsample_bytree=colsample_bytree, tree_method='hist',
                       device='cpu', n_jobs=16, random_state=seed)
        self.max_n, self.es = max_estimators, early_stopping
        self.best_iter = None
    def fit(self, v, idx, Yz, w=None, eval_set=None):
        if self.fixed_n:
            self.m = xgb.XGBRegressor(n_estimators=self.fixed_n, **self.kw)
            self.m.fit(v.X_flat[idx], Yz, sample_weight=w)
            self.best_iter = self.fixed_n
        else:
            ie, Ye, we = eval_set
            self.m = xgb.XGBRegressor(n_estimators=self.max_n, early_stopping_rounds=self.es, **self.kw)
            self.m.fit(v.X_flat[idx], Yz, sample_weight=w, eval_set=[(v.X_flat[ie], Ye)],
                       sample_weight_eval_set=[we] if we is not None else None, verbose=False)
            self.best_iter = int(self.m.best_iteration) + 1
        return self
    def predict(self, v, idx):
        it = (0, self.best_iter) if self.best_iter else None
        return v.to_q(self.m.predict(v.X_flat[idx], iteration_range=it).reshape(len(idx), -1))
