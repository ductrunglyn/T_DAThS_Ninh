"""Phân tích quan hệ thủy văn (A4): theo độ trễ, theo mùa và theo trạng thái dòng chảy.
Chỉ dùng ngày có quan trắc thật; mô tả, không dùng làm đầu vào mô hình."""
import numpy as np
import pandas as pd

FLOOD_MONTHS = [7, 8, 9, 10]
VARS = {'Q đến HN': ('hn_qden', True), 'Q ra HN': ('hn_qxa', True), 'Z HN': ('hn_mn', False),
        'Mưa V1': ('mua_v1', True), 'Mưa V2': ('mua_v2', True)}


def _prep(daily, end):
    d = daily.loc[:end].copy()
    for v in ['v1', 'v2']:
        d[f'mua_{v}_3'] = d[f'mua_{v}'].rolling(3, min_periods=3).sum()
    return d


def lag_table(daily, end, mask=None, lags=range(0, 4)):
    """corr(X(t − k), log Q CĐ(t)) cho từng biến, trên các ngày t thoả `mask` (và có quan trắc)."""
    d = _prep(daily, end)
    y = np.log1p(d.q_cd)
    base = ~d.cd_missing if mask is None else (~d.cd_missing & mask.loc[d.index])
    rows = []
    for nm, (c, lg) in VARS.items():
        x = np.log1p(d[c]) if lg else d[c]
        r = {'biến': nm}
        for k in lags:
            ok = base & x.shift(k).notna() & y.notna()
            if c.startswith('hn_'):
                ok &= ~d.hn_missing.shift(k, fill_value=True).astype(bool)
            r[f'k = {k}'] = np.corrcoef(x.shift(k)[ok], y[ok])[0, 1] if ok.sum() > 20 else np.nan
        r['n'] = int(ok.sum())
        rows.append(r)
    return pd.DataFrame(rows).set_index('biến')


def by_season_state(daily, end):
    """Tương quan theo mùa (lũ T7–T10 / kiệt) và theo trạng thái dòng chảy CĐ (thấp < Q50 ≤ trung bình < Q90 ≤ cao)."""
    d = daily.loc[:end]
    q = d.q_cd[~d.cd_missing]
    q50, q90 = q.quantile(.5), q.quantile(.9)
    flood = pd.Series(d.index.month.isin(FLOOD_MONTHS), index=d.index)
    groups = {'mùa lũ (T7–T10)': flood, 'mùa kiệt (T11–T6)': ~flood,
              f'Q CĐ thấp (< {q50:.0f})': d.q_cd < q50, f'Q CĐ trung bình ({q50:.0f}–{q90:.0f})': (d.q_cd >= q50) & (d.q_cd < q90),
              f'Q CĐ cao (≥ {q90:.0f})': d.q_cd >= q90}
    out = {g: lag_table(daily, end, m) for g, m in groups.items()}
    return pd.concat(out, names=['nhóm']), (q50, q90)


def delta_relation(daily, end, lags=range(-1, 3)):
    """corr(Δ log Q ra HN(t − k), Δ log Q CĐ(t)) theo mùa — biến động, loại xu thế mùa."""
    d = daily.loc[:end]
    dq = np.log1p(d.q_cd).diff()
    dx = np.log1p(d.hn_qxa).diff()
    ok0 = ~d.cd_missing & ~d.cd_missing.shift(1, fill_value=True) & ~d.hn_missing & ~d.hn_missing.shift(1, fill_value=True)
    flood = pd.Series(d.index.month.isin(FLOOD_MONTHS), index=d.index)
    rows = []
    for g, m in {'cả năm': pd.Series(True, index=d.index), 'mùa lũ': flood, 'mùa kiệt': ~flood}.items():
        r = {'nhóm': g}
        for k in lags:
            okk = ok0 & ok0.shift(k, fill_value=False) & m
            x = dx.shift(k)
            okk &= x.notna() & dq.notna()
            r[f'k = {k}'] = np.corrcoef(x[okk], dq[okk])[0, 1]
        rows.append(r)
    return pd.DataFrame(rows).set_index('nhóm')
