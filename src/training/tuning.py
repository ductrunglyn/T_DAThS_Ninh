"""Kiểm định chéo theo thời gian (rolling-origin, 3 fold) và dò siêu tham số — cùng một quy trình cho mọi mô hình."""
import itertools
import time

import numpy as np
import pandas as pd

from src.data.splitting import fold_names
from src.evaluation.metrics import summary
from src.training.train import fit_predict, DEEP

DETERMINISTIC = {'persistence', 'ridge'}


def seeds_for(cfg, model, kind):
    return [0] if model in DETERMINISTIC else cfg['seeds'][kind]


def run_cv(cfg, ds, model, feats, hp, seeds, gamma=0, cap=None, lam=0.0, fut=False, idx_eval_fn=None, preds=None):
    """Huấn luyện trên train của từng fold, đánh giá trên val của fold. Trả về DataFrame (fold × hạt giống).
    `preds` (list) — nếu có, nhận thêm dự báo val (ngày gốc, h, Q đo, Q dự báo) để phân tích sau."""
    rows = []
    for fold in fold_names(cfg):
        v = ds.view(fold, feats, hp.get('L', 3), fut)
        ev = idx_eval_fn(fold, v) if idx_eval_fn else v.idx_eval
        for s in seeds:
            t0 = time.time()
            P, it = fit_predict(cfg, v, model, hp, s, gamma, cap, lam, idx_eval=ev)
            rows.append({'fold': fold, 'seed': s, 'best_iter': it, 'time_s': time.time() - t0,
                         **summary(v.QO[ev], P, v.M[ev], v.q95)})
            if preds is not None:
                for h in range(P.shape[1]):
                    preds.append(pd.DataFrame({'fold': fold, 'seed': s, 'ngay_goc': v.dates[ev], 'h': h + 1,
                                               'Q_do': v.QO[ev][:, h], 'dich_that': v.M[ev][:, h], 'Q_du_bao': P[:, h]}))
    return pd.DataFrame(rows)


def cv_summary(df):
    """NSE CV = trung bình 3 fold (mỗi fold lấy trung bình các hạt giống). Tiêu chí chọn mô hình duy nhất."""
    g = df.groupby('fold').mean(numeric_only=True)
    out = {'NSE CV': g['NSE'].mean(), **{f'NSE {f}': g.loc[f, 'NSE'] for f in g.index},
           'KGE CV': g['KGE'].mean(), 'r2 CV': g['r2'].mean(), 'PBIAS CV': g['PBIAS'].mean(),
           'NSE Q>Q95 CV': g['NSE_Q95'].mean(), 'NSE CV (ĐLC giữa fold)': g['NSE'].std(ddof=1),
           'vòng tốt nhất (trung vị)': float(df['best_iter'].median())}
    return out


def grid(cfg, model):
    gd = cfg['models'][model].get('grid', {})
    keys = list(gd)
    return [dict(zip(keys, vals)) for vals in itertools.product(*[gd[k] for k in keys])] or [{'L': 3}]


def grid_search(cfg, ds, model, feats, verbose=True, L_fixed=None):
    rows, t0 = [], time.time()
    hps = grid(cfg, model)
    if L_fixed is not None:                       # thí nghiệm A: thay L của lưới bằng L cố định (bỏ trùng)
        hps = [dict(t) for t in dict.fromkeys(tuple({**h, 'L': L_fixed}.items()) for h in hps)]
    for hp in hps:
        d = run_cv(cfg, ds, model, feats, hp, seeds_for(cfg, model, 'tune'))
        rows.append({**hp, **cv_summary(d)})
        if verbose:
            print(f"{hp}: NSE CV {rows[-1]['NSE CV']:.3f}  ({time.time() - t0:.0f}s)")
    tab = pd.DataFrame(rows).sort_values('NSE CV', ascending=False).reset_index(drop=True)
    best = {k: (int(tab.loc[0, k]) if float(tab.loc[0, k]).is_integer() and k != 'max_features' else float(tab.loc[0, k]))
            for k in grid(cfg, model)[0]}
    return tab, best
