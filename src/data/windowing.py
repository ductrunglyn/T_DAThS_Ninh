"""Bộ dữ liệu chuẩn DUY NHẤT: mọi mô hình lấy dữ liệu, mẫu, vai trò train/val/test, ngưỡng Q90/Q95 và tham số chuẩn
hoá từ đây.

    ds = UnifiedDataset(cfg, H=1, max_gap=7)
    v  = ds.view('F1', feats, L=14)        # một fold (hoặc 'FINAL'), một tập biến, một độ dài cửa sổ
    v.X_seq  N × L × F   (RNN / LSTM / GRU)
    v.X_flat N × (L·F)   (Ridge-ARX / Extra Trees / XGBoost)
    v.Y      N × H       đích log(1+Q) chuẩn hoá theo train của fold
    v.M      N × H       mặt nạ ô đích có quan trắc thật
    v.QO     N × H       đích ở m³/s (NaN nếu thiếu)
    v.idx_train, v.idx_eval   chỉ số mẫu train và val/test của fold
"""
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils import path
from src.data.quality_control import build_daily
from src.data.imputation import causal_impute
from src.data.feature_engineering import build_features, FEATURE_META
from src.data.preprocessing import transform, Scaler
from src.data.splitting import make_samples, split_info


def load_daily(cfg, frozen=True):
    """Bảng ngày: ưu tiên bản ĐÓNG BĂNG data/frozen/<version>/daily_clean.csv; frozen=False → bản processed (notebook 01)."""
    from src.data.freeze import frozen_dir
    f = frozen_dir(cfg) / 'daily_clean.csv'
    if not (frozen and f.exists()):
        f = path(cfg, 'processed', 'cuadat_huana_mua_daily.csv')
    if f.exists():
        d = pd.read_csv(f, encoding='utf-8-sig', parse_dates=['date'], index_col='date')
        for c in ['cd_missing', 'hn_missing']:
            d[c] = d[c].astype(bool)
        return d
    return build_daily(cfg)[0]


@dataclass
class View:
    split: str
    feats: list
    L: int
    X_seq: np.ndarray
    Y: np.ndarray
    M: np.ndarray
    QO: np.ndarray
    W_q: np.ndarray            # Q quan trắc (0 nếu thiếu) dùng tính trọng số
    fut: np.ndarray            # mưa t+1…t+H chuẩn hoá (None nếu không dùng)
    qlast: np.ndarray          # Q đến CĐ tại ngày gốc t (đã điền nhân quả), m³/s — cho Persistence
    idx_train: np.ndarray
    idx_eval: np.ndarray
    dates: pd.DatetimeIndex
    q90: float
    q95: float
    mu_q: float
    sd_q: float

    @property
    def X_flat(self):
        x = self.X_seq.reshape(len(self.X_seq), -1)
        return np.concatenate([x, self.fut], axis=1) if self.fut is not None else x

    def to_q(self, z):
        return np.clip(np.expm1(z * self.sd_q + self.mu_q), 0, None)


class UnifiedDataset:
    def __init__(self, cfg, H, max_gap, daily=None, cache=24):
        self.cfg, self.H, self.max_gap = cfg, H, max_gap
        self.daily = load_daily(cfg) if daily is None else daily
        self.filled = causal_impute(self.daily, max_gap)
        self.F = build_features(self.filled, cfg)
        self.T = transform(self.F)
        self.index = self.T.index
        first_valid = int(np.argmax(self.T.notna().all(axis=1).values))
        self.real = ~self.daily['cd_missing'].values
        self.S, self.M = make_samples(self.index, H, first_valid, cfg['imputation']['max_window'],
                                      self.filled['unreliable'].values, self.real, cfg)
        pos = self.S['pos'].values
        q = self.daily['q_cd'].values
        self.QO = np.stack([q[t + 1:t + H + 1] for t in pos])
        self.qlast = self.filled['q_cd'].values[pos]
        self.rain_log = {v: np.log1p(self.daily[f'mua_{v}'].values) for v in ['v1', 'v2']}
        self._cache, self._cache_n = OrderedDict(), cache

    # ------------------------------------------------------------------
    def train_days(self, split):
        end, _ = split_info(self.cfg, split)
        return (self.index <= end) & ~self.filled['unreliable'].values

    def view(self, split, feats, L, fut=False):
        key = (split, tuple(feats), L, fut)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        end, ev = split_info(self.cfg, split)
        tdays = self.train_days(split)
        sc = Scaler(self.T, tdays)
        Z = sc(self.T)
        A = Z[list(feats)].values.astype(np.float32)
        pos = self.S['pos'].values
        X = A[pos[:, None] + np.arange(-L + 1, 1)[None, :]]
        assert not np.isnan(X).any(), 'cửa sổ đầu vào có giá trị rỗng'
        mu_q, sd_q = float(sc.mu['q_cd']), float(sc.sd['q_cd'])
        Y = np.where(self.M, (np.log1p(np.nan_to_num(self.QO)) - mu_q) / sd_q, 0.0).astype(np.float32)
        F = None
        if fut:
            fz = {v: (self.rain_log[v] - sc.mu[f'mua_{v}']) / sc.sd[f'mua_{v}'] for v in ['v1', 'v2']}
            F = np.stack([np.r_[fz['v1'][t + 1:t + self.H + 1], fz['v2'][t + 1:t + self.H + 1]] for t in pos]).astype(np.float32)
        qtr = self.daily['q_cd'].values[(self.index <= end) & self.real]
        role = self.S[split].values
        v = View(split=split, feats=list(feats), L=L, X_seq=X, Y=Y, M=self.M, QO=self.QO,
                 W_q=np.nan_to_num(self.QO), fut=F, qlast=self.qlast,
                 idx_train=np.where((role == 'train') & (self.S['n_real'].values > 0))[0],
                 idx_eval=np.where(role == ev)[0], dates=pd.DatetimeIndex(self.S['date']),
                 q90=float(np.quantile(qtr, .90)), q95=float(np.quantile(qtr, .95)), mu_q=mu_q, sd_q=sd_q)
        self._cache[key] = v
        if len(self._cache) > self._cache_n:
            self._cache.popitem(last=False)
        return v

    # ------------------------------------------------------------------
    def sample_table(self):
        return self.S.assign(date=self.S['date'].dt.date)

    def summary(self):
        rows = []
        for sp in [f['name'] for f in self.cfg['splits']['folds']] + [self.cfg['splits']['final']['name']]:
            end, ev = split_info(self.cfg, sp)
            r = self.S[sp].values
            tr, e = np.where(r == 'train')[0], np.where(r == ev)[0]
            qtr = self.daily['q_cd'].values[(self.index <= end) & self.real]
            rows.append({'tách': sp, 'train: số mẫu': len(tr),
                         'train: ngày gốc': f"{self.S.date.iloc[tr[0]].date()} → {self.S.date.iloc[tr[-1]].date()}",
                         'đánh giá': ev, 'đánh giá: số mẫu': len(e),
                         'đánh giá: ngày gốc': f"{self.S.date.iloc[e[0]].date()} → {self.S.date.iloc[e[-1]].date()}",
                         'Q đích max train': np.nanmax(self.QO[tr]), 'Q đích max đánh giá': np.nanmax(self.QO[e]),
                         'Q90 train': np.quantile(qtr, .9), 'Q95 train': np.quantile(qtr, .95)})
        return pd.DataFrame(rows).set_index('tách')
