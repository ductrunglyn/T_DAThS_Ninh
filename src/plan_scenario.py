"""Kịch bản bổ sung KH — giá trị của thông tin kế hoạch xả Hủa Na ngày t+1 (thiết kế: docs/ThietKe_KichBan_KeHoachXa.md).

Mọi biến thể = E2 (F1) + Q ra Hủa Na ngày t+1, ghép đúng như FUT ghép mưa tương lai (cột thêm vào X_flat; ghép vào trạng
thái ẩn cuối của mô hình hồi tiếp). Biến được ghép SAU khi View được dựng (dataclasses.replace) → không sửa src/data/,
manifest của data/frozen/v1 vẫn xác minh được.

    KH_TD        huấn luyện và dự báo bằng Q ra t+1 thực đo (cận trên)
    KH_AD{σ}     mô hình KH_TD, dự báo bằng kế hoạch có sai số σ
    KH_SS{σ}     huấn luyện và dự báo bằng kế hoạch có sai số σ
Sai số: Q̃(d) = max(0, Q(d)·(1 + σ·z_d)), z_d ~ N(0, 1), chuỗi z chung cho mọi σ, riêng cho từng hạt giống (10 000 + s).
"""
import dataclasses
import time

import numpy as np
import pandas as pd
import torch

from src.data.preprocessing import Scaler
from src.data.feature_engineering import feature_list
from src.data.splitting import fold_names
from src.training.train import fit_predict, DEEP, _t
from src.training.tuning import seeds_for
from src.evaluation.metrics import summary, peak_metrics
from src import pipeline
from src import results_store as S

SIGMAS = [0.1, 0.2, 0.3, 0.5]
MODELS = ['ridge', 'extra_trees', 'xgboost', 'rnn', 'lstm', 'gru']
NOISE_SEED0 = 10_000


def tag(kind, sigma=0.0):
    return 'KH_TD' if kind == 'TD' else f'KH_{kind}{int(round(sigma * 100))}'


def plan_series(ds, sigma=0.0, seed=0):
    """log(1 + kế hoạch xả) theo ngày (m³/s trước khi lấy log). Ngày thiếu: giá trị đã điền nhân quả (gần nhất đã biết)."""
    q = ds.filled['hn_qxa'].values.astype(float)
    if sigma:
        z = np.random.default_rng(NOISE_SEED0 + seed).standard_normal(len(q))
        q = np.clip(q * (1 + sigma * z), 0, None)
    return np.log1p(q)


def with_plan(ds, v, sigma=0.0, seed=0):
    """View v (H = 1) + cột Q ra Hủa Na ngày t+1, chuẩn hoá theo tham số của hn_qxa trên ngày train của fold."""
    assert ds.H == 1 and v.fut is None, 'kịch bản KH chỉ cho bài toán t+1, trên View không có FUT'
    sc = Scaler(ds.T, ds.train_days(v.split))
    x = (plan_series(ds, sigma, seed) - sc.mu['hn_qxa']) / sc.sd['hn_qxa']
    pos = ds.S['pos'].values
    assert pos.max() + 1 < len(x)
    return dataclasses.replace(v, fut=x[pos + 1][:, None].astype(np.float32))


def predict_with(model, obj, v, idx):
    """Dự báo lại bằng mô hình đã huấn luyện trên một View khác (cùng mẫu, khác cột kế hoạch xả)."""
    if model in DEEP:
        obj.eval()
        with torch.no_grad():
            z = obj(_t(v.X_seq[idx]), _t(v.fut[idx])).cpu().numpy()
        return v.to_q(z)
    return obj.predict(v, idx)


def _row(v, idx, P, **kw):
    return {**kw, **summary(v.QO[idx], P, v.M[idx], v.q95)}


def _pred_frame(v, idx, P, **kw):
    return pd.DataFrame({**kw, 'ngay_goc': v.dates[idx], 'h': 1, 'Q_do': v.QO[idx][:, 0],
                         'dich_that': v.M[idx][:, 0], 'Q_du_bao': P[:, 0]})


def run_cv(cfg, ds, model, verbose=True):
    """3 fold × hạt giống CV: KH_TD (+ dự báo lại với kế hoạch sai số = KH_AD), KH_SS{σ}. Trả về (bảng chỉ tiêu, dự báo)."""
    hp = pipeline.load_conf(cfg, model)['hp']
    feats = feature_list(cfg, 'F1')
    rows, preds, t0 = [], [], time.time()
    for fold in fold_names(cfg):
        v0 = ds.view(fold, feats, hp.get('L', 3))
        ev = v0.idx_eval
        # KH_TD (Ridge tất định → 1 lần) và KH_AD: cùng mô hình, dự báo với kế hoạch sai số của từng hạt giống
        objs = {}
        for s in seeds_for(cfg, model, 'cv'):
            v = with_plan(ds, v0)
            t1 = time.time()
            P, it, objs[s] = fit_predict(cfg, v, model, hp, s, return_model=True)
            rows.append(_row(v, ev, P, model=model, exp='KH_TD', fold=fold, seed=s, best_iter=it, time_s=time.time() - t1))
            preds.append(_pred_frame(v, ev, P, model=model, exp='KH_TD', fold=fold, seed=s))
        for s in cfg['seeds']['cv']:
            obj_s = objs.get(s, objs[min(objs)])   # Ridge: một mô hình KH_TD, chuỗi sai số đổi theo hạt giống
            for sg in SIGMAS:
                vn = with_plan(ds, v0, sg, s)
                P = predict_with(model, obj_s, vn, ev)
                rows.append(_row(vn, ev, P, model=model, exp=tag('AD', sg), fold=fold, seed=s, best_iter=np.nan, time_s=np.nan))
                preds.append(_pred_frame(vn, ev, P, model=model, exp=tag('AD', sg), fold=fold, seed=s))
            for sg in SIGMAS:
                vn = with_plan(ds, v0, sg, s)
                t1 = time.time()
                P, it = fit_predict(cfg, vn, model, hp, s)
                rows.append(_row(vn, ev, P, model=model, exp=tag('SS', sg), fold=fold, seed=s, best_iter=it, time_s=time.time() - t1))
                preds.append(_pred_frame(vn, ev, P, model=model, exp=tag('SS', sg), fold=fold, seed=s))
    if verbose:
        print(f'{model}: CV xong sau {time.time() - t0:.0f}s')
    return pd.DataFrame(rows), pd.concat(preds)


def run_final(cfg, ds, model, cv, peaks, verbose=True):
    """Huấn luyện cuối 2019–2024 (số vòng = trung vị vòng tốt nhất trên CV của biến thể), chấm test một lần."""
    hp = pipeline.load_conf(cfg, model)['hp']
    v0 = ds.view('FINAL', feature_list(cfg, 'F1'), hp.get('L', 3))
    te, obs = v0.idx_eval, ds.daily['q_cd']
    iterative = model in DEEP or model == 'xgboost'
    n_it = {e: (max(1, int(round(cv[cv.exp == e].best_iter.median()))) if iterative else None)
            for e in ['KH_TD'] + [tag('SS', sg) for sg in SIGMAS]}
    met, preds = [], []

    def add(exp, s, v, P, it):
        met.append({'model': model, 'exp': exp, 'seed': s, 'n_iter': it, **summary(v.QO[te], P, v.M[te], v.q95),
                    **peak_metrics(P, v.dates[te], obs, peaks, cfg['evaluation']['timing_window'])})
        preds.append(_pred_frame(v, te, P, model=model, exp=exp, seed=s))

    seeds = cfg['seeds']['final']
    for s in seeds:
        v = with_plan(ds, v0)
        P, it, obj = fit_predict(cfg, v, model, hp, s, n_iter=n_it['KH_TD'], return_model=True)
        if model not in ('ridge',) or s == seeds[0]:
            add('KH_TD', s, v, P, it)
        for sg in SIGMAS:
            vn = with_plan(ds, v0, sg, s)
            add(tag('AD', sg), s, vn, predict_with(model, obj, vn, te), it)
            Pn, itn = fit_predict(cfg, vn, model, hp, s, n_iter=n_it[tag('SS', sg)])
            add(tag('SS', sg), s, vn, Pn, itn)
    met = pd.DataFrame(met)
    if verbose:
        g = met.groupby('exp', sort=False).NSE.agg(['mean', 'std'])
        print(model, ' | '.join(f'{e} {r["mean"]:.3f}' for e, r in g.iterrows()))
    return met, pd.concat(preds)


def check_e2_reproduction(cfg, ds, model, seed=0):
    """Đường mã mới không ghép biến phải tái lập dự báo test E2 đã lưu (kiểm tra không lệch quy trình)."""
    hp = pipeline.load_conf(cfg, model)['hp']
    cv = S.read_metrics(cfg, 'cv', 1, 'chinh', model)
    d = cv[cv.exp == 'E2']
    n_it = max(1, int(round(d.best_iter.median()))) if (model in DEEP or model == 'xgboost') else None
    v = ds.view('FINAL', feature_list(cfg, 'F1'), hp.get('L', 3))
    P, _ = fit_predict(cfg, v, model, hp, seed, n_iter=n_it)
    saved = S.read_pred(cfg, 'test', 1, 'chinh', model)
    old = saved[(saved.exp == 'E2') & (saved.seed == seed) & (saved.h == 1)].set_index('ngay_goc').Q_du_bao
    new = pd.Series(P[:, 0], index=v.dates[v.idx_eval])
    assert old.index.equals(new.index)
    return float(np.abs(old.values - new.values).max())
