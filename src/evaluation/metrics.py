"""Chỉ tiêu đánh giá dùng chung. Mọi chỉ tiêu tính ở m³/s, chỉ trên ô đích có quan trắc thật.

- r²   : bình phương hệ số tương quan Pearson (KHÔNG phải hệ số xác định)
- NSE  : 1 − Σ(o − p)² / Σ(o − ō)²  (trùng với hệ số xác định R² = 1 − SSE/SST)
- KGE  : 1 − √((r − 1)² + (α − 1)² + (β − 1)²), α = σ_p/σ_o, β = μ_p/μ_o (Gupta và cs., 2009)
- PBIAS: 100 · Σ(p − o) / Σ o  (âm = dự báo thiếu)
"""
import numpy as np
import pandas as pd

BASIC = ['MSE', 'MAE', 'RMSE', 'r2', 'NSE', 'KGE', 'PBIAS', 'UNDER']
LABEL = {'MSE': 'MSE', 'MAE': 'MAE (m³/s)', 'RMSE': 'RMSE (m³/s)', 'r2': 'r²', 'NSE': 'NSE', 'KGE': 'KGE',
         'PBIAS': 'PBIAS (%)', 'NSE_Q95': 'NSE Q>Q95', 'RMSE_Q95': 'RMSE Q>Q95', 'Peak_RMSE': 'Peak RMSE',
         'Peak_Bias': 'Peak Bias (%)', 'Peak_Timing': 'Sai số thời điểm đỉnh (ngày)',
         'UNDER': 'Dự báo thiếu TB (m³/s)', 'UNDER_Q95': 'Dự báo thiếu TB Q>Q95 (m³/s)'}
DIGITS = {'MSE': 0, 'MAE': 1, 'RMSE': 1, 'r2': 3, 'NSE': 3, 'KGE': 3, 'PBIAS': 1, 'NSE_Q95': 3, 'RMSE_Q95': 1,
          'Peak_RMSE': 0, 'Peak_Bias': 1, 'Peak_Timing': 2, 'UNDER': 1, 'UNDER_Q95': 1}


def basic(o, p):
    o, p = np.asarray(o, float), np.asarray(p, float)
    e = p - o
    mse = (e ** 2).mean()
    r = np.corrcoef(o, p)[0, 1] if o.std() > 0 and p.std() > 0 else 0.0
    alpha = p.std() / o.std() if o.std() > 0 else np.nan
    beta = p.mean() / o.mean() if o.mean() != 0 else np.nan
    return {'MSE': mse, 'MAE': np.abs(e).mean(), 'RMSE': np.sqrt(mse), 'r2': r ** 2,
            'NSE': 1 - (e ** 2).sum() / ((o - o.mean()) ** 2).sum(),
            'KGE': 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2),
            'PBIAS': 100 * e.sum() / o.sum(),
            'UNDER': float(np.maximum(o - p, 0).mean())}           # sai số dự báo thiếu trung bình


def per_horizon(O, P, M):
    return pd.DataFrame([{'h': h + 1, 'n': int(M[:, h].sum()), **basic(O[M[:, h], h], P[M[:, h], h])}
                         for h in range(O.shape[1])]).set_index('h')


def summary(O, P, M, q95=None):
    """Chỉ tiêu tổng hợp: trung bình theo h cho các chỉ tiêu cơ bản; nhóm lưu lượng cao gộp mọi h."""
    r = per_horizon(O, P, M)[BASIC].mean().to_dict()
    if q95 is not None:
        hi = M & (np.nan_to_num(O) > q95)
        if hi.sum() > 5:
            o, p = O[hi], P[hi]
            r['NSE_Q95'] = 1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum()
            r['RMSE_Q95'] = float(np.sqrt(((o - p) ** 2).mean()))
            r['UNDER_Q95'] = float(np.maximum(o - p, 0).mean())
        else:
            r['NSE_Q95'] = r['RMSE_Q95'] = r['UNDER_Q95'] = np.nan
    return r


# ---------------------------------------------------------------- đỉnh lũ
def find_peaks(obs, threshold, halfwidth=7):
    """Đỉnh độc lập: giá trị lớn nhất trong ±halfwidth ngày và > threshold."""
    s = obs.dropna()
    rm = s.rolling(2 * halfwidth + 1, center=True, min_periods=1).max()
    return list(s.index[(s == rm) & (s > threshold)])


def peak_metrics(P, origins, obs, peaks, timing_window=3):
    """P: [n, H] dự báo theo ngày gốc `origins`; obs: chuỗi quan trắc theo ngày.
    Peak RMSE / Peak Bias: tại ngày đỉnh, mọi h có ngày gốc hợp lệ; Peak_Timing: |ngày cực đại của dự báo h = 1 trong
    ±timing_window ngày quanh đỉnh − ngày đỉnh| (trung bình)."""
    pos = {d: i for i, d in enumerate(origins)}
    H = P.shape[1]
    err, rel = [], []
    for d in peaks:
        for h in range(1, H + 1):
            o = d - pd.Timedelta(days=h)
            if o in pos:
                e = P[pos[o], h - 1] - obs[d]
                err.append(e); rel.append(100 * e / obs[d])
    s1 = pd.Series(P[:, 0], index=pd.DatetimeIndex(origins) + pd.Timedelta(days=1))
    tim = []
    for d in peaks:
        w = s1.loc[d - pd.Timedelta(days=timing_window):d + pd.Timedelta(days=timing_window)]
        if len(w):
            tim.append(abs((w.idxmax() - d).days))
    return {'Peak_RMSE': float(np.sqrt(np.mean(np.square(err)))) if err else np.nan,
            'Peak_Bias': float(np.mean(rel)) if rel else np.nan,
            'Peak_Timing': float(np.mean(tim)) if tim else np.nan}
