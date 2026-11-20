"""Chạy RIÊNG một mô hình × một kịch bản — dùng chung cho notebook 10 và experiments/chay_mo_hinh.py.

Dùng đúng các hàm của src/pipeline.py (cùng dữ liệu đóng băng, cùng hạt giống, cùng cách chọn số vòng), nên kết quả trùng
với kết quả chính trên cùng phần cứng. train_one() KHÔNG ghi file kết quả (trả về bảng để xem ngay); chỉ khi
save_model=True mới lưu weight vào results/chay_le/ — không bao giờ ghi đè kết quả chính.

    saved_results(cfg, model, exp, H)   kết quả đã lưu (đọc results/ket_qua_thuc_nghiem.csv, không tính lại)
    test_saved(cfg, model, exp, H)      chỉ chấm tập kiểm tra bằng weight tốt nhất đã lưu (E2, t+1), không huấn luyện
    train_one(cfg, model, exp, H)       CV 3 fold → huấn luyện cuối 2019–2024 → chấm tập kiểm tra
"""
import json
import re
import time
import warnings

import numpy as np
import pandas as pd

from src.utils import results_dir
from src import pipeline, plan_scenario as K
from src import results_store as S
from src.data.feature_engineering import feature_list
from src.data.splitting import fold_names
from src.training.train import fit_predict, DEEP, DISPLAY, MODEL_NAMES
from src.training.tuning import seeds_for, cv_summary, grid_search
from src.evaluation.metrics import summary, peak_metrics

warnings.filterwarnings('ignore', message='Mean of empty slice')

BASE = {'E1': 'F0', 'E2': 'F1', 'E3': 'F2', 'E4': 'F3', 'P_F4': 'F4', 'P_F5': 'F5', 'P_EMA': 'F1_EMA'}
EXPS = ['PERS'] + list(BASE) + ['E5', 'E6', 'E56', 'FUT', 'KH_TD'] + \
       [f'KH_{k}{s}' for k in ('AD', 'SS') for s in (10, 20, 30, 50)]
SHOW = ['NSE', 'KGE', 'RMSE', 'MAE', 'PBIAS', 'NSE_Q95', 'Peak_Bias', 'Peak_Timing']


def describe(exp):
    m = re.fullmatch(r'KH_(TD|AD|SS)(\d*)', exp)
    if not m:
        return pipeline.EXP_NAME.get(exp, exp)
    return {'TD': 'E2 + Q ra Hủa Na t+1 thực đo (cận trên)',
            'AD': f'mô hình KH_TD, dự báo bằng kế hoạch xả sai số {m.group(2)} %',
            'SS': f'huấn luyện và dự báo bằng kế hoạch xả sai số {m.group(2)} %'}[m.group(1)]


# ---------------------------------------------------------------------------------------------------------------- kịch bản
def spec(exp, model, conf, H):
    """Kịch bản → tập biến, trọng số (E5), λ (E6), FUT, kế hoạch xả (KH). Giống hệt cách pipeline / notebook 09 đặt."""
    if exp not in EXPS:
        raise ValueError(f'kịch bản không hợp lệ: {exp}. Chọn trong {EXPS}')
    if model == 'persistence' and exp != 'PERS':
        raise ValueError('Persistence chỉ có kịch bản PERS')
    if exp == 'PERS':
        if model != 'persistence':
            raise ValueError('PERS chỉ dùng cho Persistence')
        return dict(fs='F0')
    if exp in BASE:
        return dict(fs=BASE[exp])
    sp = dict(fs='F1')
    if exp in ('E5', 'E56'):
        sp.update(gamma=conf['weights']['gamma'], cap=conf['weights']['cap'])
    if exp in ('E6', 'E56'):
        if model not in DEEP:
            raise ValueError('E6, E56 (phạt dự báo thiếu) chỉ dùng cho RNN, LSTM, GRU')
        sp['lam'] = conf['lambda']
    if exp == 'FUT':
        sp['fut'] = True
    m = re.fullmatch(r'KH_(TD|AD|SS)(\d*)', exp)
    if m:
        if H != 1:
            raise ValueError('kịch bản KH chỉ cho bài toán t+1 (H = 1)')
        sg = int(m.group(2) or 0) / 100
        sp.update(kh=m.group(1), sigma_train=0.0 if m.group(1) in ('TD', 'AD') else sg, sigma_eval=sg)
    return sp


def _views(ds, split, sp, hp, seed):
    """(View huấn luyện, View dự báo). Khác nhau chỉ ở KH_AD: huấn luyện bằng Q ra thực đo, dự báo bằng kế hoạch sai số."""
    v = ds.view(split, feature_list(ds.cfg, sp['fs']), hp.get('L', 3), sp.get('fut', False))
    if 'kh' not in sp:
        return v, v
    return K.with_plan(ds, v, sp['sigma_train'], seed), K.with_plan(ds, v, sp['sigma_eval'], seed)


def _seeds(cfg, model, sp, kind):
    """Hạt giống như pipeline; riêng KH_AD / KH_SS: mọi mô hình dùng đủ hạt giống (mỗi hạt giống một chuỗi sai số)."""
    if sp.get('kh') in ('AD', 'SS'):
        return cfg['seeds'][kind]
    return seeds_for(cfg, model, kind)


def _fit_eval(cfg, model, vt, ve, hp, sp, seed, n_iter=None):
    P, it, obj = fit_predict(cfg, vt, model, hp, seed, sp.get('gamma', 0), sp.get('cap'), sp.get('lam', 0.0),
                             n_iter=n_iter, return_model=True)
    if ve is not vt:                                      # KH_AD: dự báo lại với kế hoạch có sai số
        P = K.predict_with(model, obj, ve, vt.idx_eval)
    return P, it, obj


# ---------------------------------------------------------------------------------------------------------------- cấu hình
def configure(cfg, model, exp, tu_dau=False, log=print):
    """Siêu tham số, trọng số E5, λ E6: đọc config.json đã chọn (mặc định) hoặc chọn lại từ đầu bằng CV (tu_dau=True)."""
    if not tu_dau:
        conf = pipeline.load_conf(cfg, model)
        if not conf.get('hp'):
            raise FileNotFoundError(f'chưa có cấu hình của {model} trong results/cau_hinh_chon.json — chạy notebook của '
                                    f'mô hình trước, hoặc dùng tu_dau=True')
        return conf
    ds = pipeline.get_dataset(cfg, 1)
    f1 = feature_list(cfg, 'F1')
    conf = {}
    if model == 'persistence':
        conf['hp'] = {'L': 3}
    else:
        log('Dò siêu tham số trên E2, t+1, CV 3 fold …')
        tab, conf['hp'] = grid_search(cfg, ds, model, f1, verbose=False)
        log(f"  chọn {conf['hp']} (NSE CV {tab['NSE CV'].iloc[0]:.3f})")
    if exp in ('E5', 'E56'):
        res = {}
        for w in cfg['weights']['settings']:
            d = pipeline.run_cv(cfg, ds, model, f1, conf['hp'], seeds_for(cfg, model, 'cv'), w['gamma'], w['cap'])
            res[(w['gamma'], w['cap'])] = cv_summary(d)['NSE CV']
        g, c = max(res, key=res.get)
        conf['weights'] = {'gamma': g, 'cap': c}
        log(f'  E5: chọn γ = {g}, chặn trên = {c} theo NSE CV')
    if exp in ('E6', 'E56'):
        res = {lam: cv_summary(pipeline.run_cv(cfg, ds, model, f1, conf['hp'], seeds_for(cfg, model, 'cv'), lam=lam))['NSE CV']
               for lam in cfg['penalty']['lambda']}
        conf['lambda'] = float(max(res, key=res.get))
        log(f"  E6: chọn λ = {conf['lambda']} theo NSE CV")
    return conf


# ---------------------------------------------------------------------------------------------------------------- huấn luyện
def train_one(cfg, model, exp, H=1, tu_dau=False, save_model=False, log=print):
    """Huấn luyện từ đầu tới test cho MỘT mô hình × MỘT kịch bản.
    Trả về dict: cv (theo fold × hạt giống), cv_tom_tat, test (theo hạt giống), pred (dự báo test), n_iter, conf,
    so_voi_ket_qua_chinh (chênh NSE lớn nhất, None nếu chưa có). Không ghi file, trừ weight khi save_model=True."""
    say = log

    t0 = time.time()
    conf = configure(cfg, model, exp, tu_dau, say)
    hp = conf['hp']
    sp = spec(exp, model, conf, H)
    ds = pipeline.get_dataset(cfg, H)
    say(f'{DISPLAY[model]} · {exp} ({describe(exp)}) · H = {H} · siêu tham số {hp}')

    # 1. CV 3 fold: chỉ tiêu trên tập xác thực và số vòng tốt nhất (dừng sớm theo tập xác thực của fold)
    cv_seeds = _seeds(cfg, model, sp, 'cv')
    if H != 1:
        cv_seeds = cv_seeds[:2]                           # như pipeline.extended()
    rows = []
    for fold in fold_names(cfg):
        for s in cv_seeds:
            vt, ve = _views(ds, fold, sp, hp, s)
            P, it, _ = _fit_eval(cfg, model, vt, ve, hp, sp, s)
            rows.append({'fold': fold, 'seed': s, 'best_iter': it, **summary(ve.QO[ve.idx_eval], P, ve.M[ve.idx_eval], ve.q95)})
    cv = pd.DataFrame(rows)
    s_cv = cv_summary(cv)
    say(f"CV: NSE {s_cv['NSE CV']:.3f} (F1 {s_cv['NSE F1']:.3f} · F2 {s_cv['NSE F2']:.3f} · F3 {s_cv['NSE F3']:.3f}) "
        f"· NSE Q>Q95 {s_cv['NSE Q>Q95 CV']:.3f}")

    # 2. Huấn luyện cuối 2019–2024, số vòng = trung vị vòng tốt nhất trên CV; 3. chấm tập kiểm tra
    iterative = model in DEEP or model == 'xgboost'
    n_iter = max(1, int(round(cv['best_iter'].median()))) if iterative else None
    peaks, _ = pipeline.test_peaks(cfg, ds)
    obs = ds.daily['q_cd']
    met, preds = [], []
    for s in _seeds(cfg, model, sp, 'final'):
        vt, ve = _views(ds, 'FINAL', sp, hp, s)
        te = ve.idx_eval
        P, it, obj = _fit_eval(cfg, model, vt, ve, hp, sp, s, n_iter=n_iter)
        met.append({'model': model, 'exp': exp, 'seed': s, 'n_iter': it, **summary(ve.QO[te], P, ve.M[te], ve.q95),
                    **peak_metrics(P, ve.dates[te], obs, peaks, cfg['evaluation']['timing_window'])})
        for h in range(P.shape[1]):
            preds.append(pd.DataFrame({'model': model, 'exp': exp, 'seed': s, 'ngay_goc': ve.dates[te], 'h': h + 1,
                                       'Q_do': ve.QO[te][:, h], 'dich_that': ve.M[te][:, h], 'Q_du_bao': P[:, h]}))
        if save_model:
            _save_model(results_dir(cfg, 'chay_le', model, f'{exp}_H{H}'), model, obj, s)
    met, pred = pd.DataFrame(met), pd.concat(preds)
    say(f'Huấn luyện cuối: {len(met)} hạt giống, số vòng {n_iter}. Tập kiểm tra: NSE {met.NSE.mean():.3f}'
        + (f' ± {met.NSE.std(ddof=1):.3f}' if len(met) > 1 else ''))

    diff = compare_saved(cfg, model, exp, H, met)
    if diff is not None:
        say(f"Đối chiếu với kết quả chính đã lưu: chênh NSE lớn nhất {diff:.2e} "
            f"({'trùng' if diff < 1e-3 else 'KHÁC — do tu_dau, đổi cấu hình, hạt giống hoặc phần cứng'})")
    say(f'Xong sau {time.time() - t0:.0f} s.' + (f' Weight: results/chay_le/{model}/{exp}_H{H}/' if save_model else ''))
    return {'cv': cv, 'cv_tom_tat': s_cv, 'test': met, 'pred': pred, 'n_iter': n_iter, 'conf': conf,
            'so_voi_ket_qua_chinh': diff}


def _save_model(out, model, obj, s):
    if model in DEEP:
        import torch
        torch.save(obj.state_dict(), out / f'seed{s}.pt')
    elif model != 'persistence':
        import joblib
        joblib.dump(obj, out / f'seed{s}.joblib')


def compare_saved(cfg, model, exp, H, met):
    """Chênh NSE lớn nhất (theo hạt giống) giữa lần chạy này và kết quả chính đã lưu; None nếu chưa có."""
    old = S.read_metrics(cfg, 'test', H, 'KH' if exp.startswith('KH') else 'chinh', model, exp)
    if not len(old):
        return None
    j = met.merge(old[['seed', 'NSE']], on='seed', suffixes=('', '_luu'))
    return float(np.abs(j.NSE - j.NSE_luu).max()) if len(j) else None


# ---------------------------------------------------------------------------------------------------------------- chỉ test
def saved_models(cfg, model):
    """Mô tả weight tốt nhất đã lưu của mô hình: {kịch bản: mô tả} (đọc final_models/mo_hinh.json)."""
    f = pipeline.out_dir(cfg, model) / 'final_models' / 'mo_hinh.json'
    return json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}


def test_saved(cfg, model, exp='E2', H=1):
    """Nạp weight tốt nhất đã lưu → dự báo tập kiểm tra (CPU) → chỉ tiêu từng hạt giống. Không huấn luyện.
    Trả về (bảng chỉ tiêu, dự báo test). Weight đã lưu: E2, t+1, 5 hạt giống (Ridge-ARX tất định: 1)."""
    import joblib
    import torch
    allm = saved_models(cfg, model)
    meta = allm.get(exp)
    if meta is None or meta['H'] != H:
        have = ', '.join(f'{e} (H = {x["H"]})' for e, x in allm.items()) or 'không có'
        raise FileNotFoundError(f'không có weight đã lưu cho {model} {exp} H = {H}. Đã lưu: {have}. Kịch bản khác: xem '
                                f'saved_results() hoặc huấn luyện lại bằng train_one().')
    d = pipeline.out_dir(cfg, model) / 'final_models'
    ds = pipeline.get_dataset(cfg, H)
    peaks, _ = pipeline.test_peaks(cfg, ds)
    saved = S.read_pred(cfg, 'test', H, 'chinh', model, exp)
    rows, preds = [], []
    for seed, fname in meta['files'].items():
        f = d / fname
        meta['seed'] = int(seed)
        v = ds.view('FINAL', meta['feats'], meta['L'], meta['fut'])
        te = v.idx_eval
        if meta['model'] in DEEP:
            net = DEEP[meta['model']].build(len(meta['feats']), meta['hp']['hidden'], meta['hp']['layers'],
                                            cfg['deep']['dropout'], meta['H'], v.fut.shape[1] if v.fut is not None else 0)
            net.load_state_dict(torch.load(f, map_location='cpu'))
            net.eval()
            with torch.no_grad():
                P = v.to_q(net(torch.as_tensor(v.X_seq[te]),
                               torch.as_tensor(v.fut[te]) if v.fut is not None else None).numpy())
        else:
            P = joblib.load(f).predict(v, te)
        s = saved[saved.seed == meta['seed']].pivot(index='ngay_goc', columns='h', values='Q_du_bao')
        rows.append({'seed': meta['seed'], **summary(v.QO[te], P, v.M[te], v.q95),
                     **peak_metrics(P, v.dates[te], ds.daily['q_cd'], peaks, cfg['evaluation']['timing_window']),
                     'chênh so với dự báo đã lưu (m³/s)': float(np.abs(P - s.values).max())})
        for h in range(P.shape[1]):
            preds.append(pd.DataFrame({'model': model, 'exp': exp, 'seed': meta['seed'], 'ngay_goc': v.dates[te], 'h': h + 1,
                                       'Q_do': v.QO[te][:, h], 'dich_that': v.M[te][:, h], 'Q_du_bao': P[:, h]}))
    return pd.DataFrame(rows).set_index('seed'), pd.concat(preds)


# ---------------------------------------------------------------------------------------------------------------- đã lưu
def saved_results(cfg, model=None, exp=None, H=1):
    """Bảng kết quả đã lưu (results/ket_qua_thuc_nghiem.csv): NSE CV (tiêu chí lựa chọn) và trung bình các hạt giống trên
    tập kiểm tra, cho mọi kịch bản (gồm kịch bản KH khi H = 1)."""
    models = [model] if model else S.models_with(cfg, 'test', H)
    nhoms = ['chinh', 'KH'] if H == 1 else ['chinh']
    met = pd.concat([S.read_metrics(cfg, 'test', H, n, m) for n in nhoms for m in models])
    if not len(met):
        raise FileNotFoundError('chưa có kết quả — chạy các notebook 03–06 (và 09) trước')
    if exp:
        met = met[met.exp == exp]
    cvs = [{'model': m, 'exp': e, 'NSE CV': cv_summary(g)['NSE CV']}
           for n in nhoms for m in models for e, g in S.read_metrics(cfg, 'cv', H, n, m).groupby('exp', sort=False)]
    order = {e: i for i, e in enumerate(EXPS)}
    met = met.assign(_m=met.model.map(MODEL_NAMES.index), _e=met.exp.map(lambda e: order.get(e, 99))).sort_values(['_m', '_e'])
    g = met.groupby(['model', 'exp'], sort=False)
    t = g[SHOW].mean()
    t.insert(0, 'số hạt giống', g.size())
    t.insert(1, 'ĐLC NSE', g.NSE.std(ddof=1))
    if cvs:
        t.insert(0, 'NSE CV', pd.DataFrame(cvs).set_index(['model', 'exp'])['NSE CV'].reindex(t.index))
    t.index = pd.MultiIndex.from_tuples([(DISPLAY[m], e) for m, e in t.index], names=['mô hình', 'kịch bản'])
    return t


def mean_series(pred, h=1):
    """Dự báo trung bình các hạt giống và quan trắc theo NGÀY ĐÍCH, cho bước h (để vẽ)."""
    d = pred[pred.h == h]
    p = d.groupby('ngay_goc').Q_du_bao.mean()
    o = d.groupby('ngay_goc').Q_do.first()
    idx = pd.DatetimeIndex(p.index) + pd.Timedelta(days=h)
    return pd.Series(o.values, index=idx, name='quan trắc'), pd.Series(p.values, index=idx, name='dự báo')
