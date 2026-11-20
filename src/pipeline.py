"""Quy trình thực nghiệm chung cho MỌI mô hình (notebook mô hình chỉ gọi các hàm ở đây).

Bài toán chính H = 1 (Q_CĐ(t+1)), mở rộng H = 10.
  1. tune()            dò siêu tham số trên F1 (E2), hàm mất mát cơ bản, 3 fold
  2. cv_experiments()  E1–E4, thực nghiệm phụ, E5 (trọng số), E6 (phạt dự báo thiếu), cận trên mưa tương lai
  3. final_test()      huấn luyện lại 2019–2024, dự báo test 01/2025–08/2026 (KHOÁ: không dùng để chọn gì)
  4. extended()        H = 10 cho E1, E2, E5

Kết quả ghi vào các file GỘP (src/results_store.py): results/ket_qua_thuc_nghiem.csv (chỉ tiêu CV + test),
results/du_bao.csv.gz (dự báo theo ngày), results/do_sieu_tham_so.csv, results/cau_hinh_chon.json.
Thư mục results/<baselines|models>/<mô hình>/ chỉ còn weight tốt nhất (final_models/: E2, t+1) và nhật ký huấn luyện.
"""
import json
import time

import numpy as np
import pandas as pd

from src.utils import path
from src import results_store as S
from src.data.windowing import UnifiedDataset
from src.data.feature_engineering import feature_list
from src.data.splitting import fold_names
from src.training.train import fit_predict, DEEP, DISPLAY
from src.training.tuning import run_cv, cv_summary, grid_search, seeds_for
from src.evaluation.metrics import summary, find_peaks, peak_metrics

BASELINES = ['persistence', 'ridge', 'extra_trees', 'xgboost']
BASE_EXPS = [('E1', 'F0', 'Không Hủa Na'), ('E2', 'F1', 'Chỉ Q ra Hủa Na'), ('E3', 'F2', 'Chỉ Q đến Hủa Na'),
             ('E4', 'F3', 'Q ra + Q đến + Z Hủa Na'), ('P_F4', 'F4', 'Phụ: đủ 6 biến Hủa Na'),
             ('P_F5', 'F5', 'Phụ: E4 + đặc trưng thủy văn (ΔZ, cân bằng, tỉ lệ xả)'),
             ('P_EMA', 'F1_EMA', 'Phụ: E2 + trung bình trượt mũ Q CĐ')]
EXP_NAME = {e: n for e, _, n in BASE_EXPS}
EXP_NAME.update({'E5': 'E2 + hàm mất mát có trọng số (chọn theo CV)', 'E6': 'E2 + phạt dự báo thiếu (chọn theo CV)',
                 'E56': 'E2 + trọng số + phạt dự báo thiếu', 'FUT': 'E2 + mưa thực đo t+1 (cận trên)',
                 'PERS': 'Persistence'})


def out_dir(cfg, model):
    """Thư mục riêng của mô hình: weight tốt nhất (final_models/) và nhật ký huấn luyện. Không tự tạo."""
    return path(cfg, 'results', 'baselines' if model in BASELINES else 'models', model)


def load_conf(cfg, model, variant=None):
    """Cấu hình chọn theo CV (results/cau_hinh_chon.json). variant='L3': thí nghiệm A."""
    return S.load_conf(cfg, model, S.nhom_of(variant))


def save_conf(cfg, model, conf, variant=None):
    S.save_conf(cfg, model, conf, S.nhom_of(variant))


def get_dataset(cfg, H):
    """Bộ dữ liệu cho nhóm mô hình: CHỈ từ bản đóng băng (kiểm tra SHA-256 + tập mẫu trùng khớp)."""
    from src.data.freeze import verify, frozen_dir
    man = verify(cfg)
    ds = UnifiedDataset(cfg, H, man['max_gap'])
    fr = pd.read_csv(frozen_dir(cfg) / f'samples_H{H}.csv', parse_dates=['date'])
    cur = ds.sample_table()
    assert (fr['pos'].values == cur['pos'].values).all() and (fr['reliable'].values == cur['reliable'].values).all() \
        and all((fr[c].fillna('').values == cur[c].fillna('').values).all() for c in fr.columns if c[0] in 'FT'), \
        'tập mẫu dựng lại khác bản đóng băng'
    return ds


# ---------------------------------------------------------------- 0. chọn ngưỡng khoảng thiếu (notebook 02)
def select_gap(cfg, verbose=True):
    gs = cfg['gap_selection']
    feats = feature_list(cfg, gs['feature_set'])
    from src.data.windowing import load_daily
    daily = load_daily(cfg, frozen=False)
    dss = {G: UnifiedDataset(cfg, cfg['horizons']['main'], G, daily) for G in cfg['imputation']['gap_candidates']}
    # đánh giá trên các mẫu val hợp lệ với MỌI ngưỡng (giao) để so sánh công bằng
    common = {f: np.logical_and.reduce([d.S[f].values == 'val' for d in dss.values()]) for f in fold_names(cfg)}
    rows = []
    for G, ds in dss.items():
        for m in gs['models']:
            d = run_cv(cfg, ds, m, feats, gs['hp'], cfg['seeds']['tune'],
                       idx_eval_fn=lambda fold, v: np.where(common[fold])[0])
            s = cv_summary(d)
            n_tr = {f: int((ds.S[f] == 'train').sum()) for f in fold_names(cfg)}
            rows.append({'ngưỡng G (ngày)': G, 'mô hình': DISPLAY[m], **s, **{f'số mẫu train {f}': n for f, n in n_tr.items()}})
            if verbose:
                print(f"G = {G}, {DISPLAY[m]}: NSE CV {s['NSE CV']:.3f}")
    tab = pd.DataFrame(rows)
    mean = tab.groupby('ngưỡng G (ngày)')['NSE CV'].mean()
    return tab, int(mean.idxmax()), mean


# ---------------------------------------------------------------- 1. dò siêu tham số
def tune(cfg, model, verbose=True, variant=None, L_fixed=None):
    ds = get_dataset(cfg, cfg['horizons']['main'])
    feats = feature_list(cfg, 'F1')
    if model == 'persistence':
        tab, best = pd.DataFrame([{'L': 3}]), {'L': 3}
    else:
        tab, best = grid_search(cfg, ds, model, feats, verbose, L_fixed)
    S.write_tuning(cfg, tab, model, S.nhom_of(variant))
    conf = load_conf(cfg, model, variant)
    conf.update({'model': model, 'hp': best, 'max_gap': ds.max_gap, 'dataset_version': cfg['dataset_version']})
    save_conf(cfg, model, conf, variant)
    return tab, best


# ---------------------------------------------------------------- 2. thực nghiệm CV
def _specs_base(model):
    if model == 'persistence':
        return [dict(exp='PERS', fs='F0')]
    return [dict(exp=e, fs=fs) for e, fs, _ in BASE_EXPS]


def cv_experiments(cfg, model, verbose=True, variant=None):
    conf = load_conf(cfg, model, variant)
    hp = conf['hp']
    ds = get_dataset(cfg, cfg['horizons']['main'])
    seeds = seeds_for(cfg, model, 'cv')
    rows, preds, t0 = [], [], time.time()

    def run(spec):
        pl = []
        d = run_cv(cfg, ds, model, feature_list(cfg, spec['fs']), hp, seeds, spec.get('gamma', 0), spec.get('cap'),
                   spec.get('lam', 0.0), spec.get('fut', False), preds=pl)
        preds.append(pd.concat(pl).assign(exp=spec['exp']))
        for k, x in spec.items():
            d[k] = x if not isinstance(x, (list, dict)) else str(x)
        rows.append(d)
        s = cv_summary(d)
        if verbose:
            print(f"{spec['exp']:12s} NSE CV {s['NSE CV']:.3f} | NSE Q>Q95 {s['NSE Q>Q95 CV']:.3f}  ({time.time() - t0:.0f}s)")
        return s

    for sp in _specs_base(model):
        run(sp)
    if model != 'persistence':
        res = {}
        for w in cfg['weights']['settings']:
            tag = f"W_g{w['gamma']}_c{w['cap'] or 'none'}"
            res[tag] = (run(dict(exp=tag, fs='F1', gamma=w['gamma'], cap=w['cap'])), w)
        tag = max(res, key=lambda k: res[k][0]['NSE CV'])
        conf['weights'] = res[tag][1]
        run(dict(exp='E5', fs='F1', gamma=conf['weights']['gamma'], cap=conf['weights']['cap']))
        if model in ('lstm', 'gru'):
            pres = {lam: run(dict(exp=f'PEN_l{lam}', fs='F1', lam=lam)) for lam in cfg['penalty']['lambda']}
            conf['lambda'] = float(max(pres, key=lambda k: pres[k]['NSE CV']))
            run(dict(exp='E6', fs='F1', lam=conf['lambda']))
            run(dict(exp='E56', fs='F1', gamma=conf['weights']['gamma'], cap=conf['weights']['cap'], lam=conf['lambda']))
        run(dict(exp='FUT', fs='F1', fut=True))
    cv = pd.concat(rows, ignore_index=True)
    cv.insert(0, 'model', model)
    S.write_metrics(cfg, cv, model, 1, 'cv', S.nhom_of(variant))
    S.write_pred(cfg, pd.concat(preds), model, 1, 'cv', S.nhom_of(variant))
    save_conf(cfg, model, conf, variant)
    return cv_table(cv)


def cv_table(cv):
    rows = []
    for e, d in cv.groupby('exp', sort=False):
        rows.append({'thực nghiệm': e, 'mô tả': EXP_NAME.get(e, e), 'tập biến': d['fs'].iloc[0], **cv_summary(d)})
    return pd.DataFrame(rows).set_index('thực nghiệm')


# ---------------------------------------------------------------- 3. huấn luyện cuối + test (khoá)
FINAL_EXPS = ['E1', 'E2', 'E3', 'E4', 'P_F4', 'P_F5', 'P_EMA', 'E5', 'E6', 'E56', 'FUT', 'PERS']


def test_peaks(cfg, ds):
    fn = cfg['splits']['final']
    q = ds.daily['q_cd']
    thr = float(q[(q.index <= fn['train_end']) & ~ds.daily['cd_missing']].quantile(cfg['evaluation']['peak_quantile']))
    peaks = find_peaks(q[q.index >= fn['test_start'] + pd.Timedelta(days=1)], thr, cfg['evaluation']['peak_halfwidth'])
    return peaks, thr


SAVE_MODELS = {'H1': ['E2']}      # weight tốt nhất được lưu: cấu hình chính E2, t+1 (5 hạt giống); dùng cho "chỉ test" và
                                  # tests/test_final_models.py. Các kịch bản khác chỉ lưu chỉ tiêu và dự báo.


def _final(cfg, model, H, exps, cv, tag, verbose=True, variant=None):
    import joblib, torch
    conf = load_conf(cfg, model, variant)
    md_dir = out_dir(cfg, model) / 'final_models'
    save = [e for e in exps if e in SAVE_MODELS.get(tag, [])] if (variant is None and model != 'persistence') else []
    if save and md_dir.exists():                       # thay toàn bộ weight cũ bằng lần huấn luyện này
        for f in md_dir.iterdir():
            f.unlink()
    meta = {}
    logs = []
    hp = conf['hp']
    ds = get_dataset(cfg, H)
    peaks, _ = test_peaks(cfg, ds)
    obs = ds.daily['q_cd']
    seeds = seeds_for(cfg, model, 'final')
    pred_rows, met_rows = [], []
    for e in exps:
        d = cv[cv.exp == e]
        if not len(d):
            continue
        sp = d.iloc[0]
        gamma = int(sp['gamma']) if 'gamma' in d and pd.notna(sp.get('gamma')) else 0
        cap = sp['cap'] if 'cap' in d and pd.notna(sp.get('cap')) else None
        lam = float(sp['lam']) if 'lam' in d and pd.notna(sp.get('lam')) else 0.0
        fut = bool(sp['fut']) if 'fut' in d and pd.notna(sp.get('fut')) else False
        n_iter = max(1, int(round(d['best_iter'].median()))) if (model in DEEP or model == 'xgboost') else None
        v = ds.view('FINAL', feature_list(cfg, sp['fs']), hp.get('L', 3), fut)
        te = v.idx_eval
        for s in seeds:
            P, it, mobj = fit_predict(cfg, v, model, hp, s, gamma, cap, lam, n_iter=n_iter, return_model=True)
            for rec in getattr(mobj, 'train_log', []):
                logs.append({'exp': e, 'seed': s, **rec})
            if e in save:
                md_dir.mkdir(parents=True, exist_ok=True)
                f = md_dir / f'{tag}_{e}_seed{s}.{"pt" if model in DEEP else "joblib"}'
                if model in DEEP:
                    torch.save(mobj.state_dict(), f)
                else:
                    joblib.dump(mobj, f)
                m_ = meta.setdefault(e, {'model': model, 'exp': e, 'feature_set': sp['fs'], 'feats': feature_list(cfg, sp['fs']),
                                         'hp': hp, 'H': H, 'L': hp.get('L', 3), 'fut': fut, 'gamma': gamma, 'cap': cap,
                                         'lambda': lam, 'n_iter': n_iter, 'dataset_version': cfg['dataset_version'],
                                         'max_gap': ds.max_gap, 'train_end': str(cfg['splits']['final']['train_end'].date()),
                                         'q_train_max': float(np.nanmax(v.QO[v.idx_train])), 'files': {}})
                m_['files'][str(s)] = f.name
            met_rows.append({'model': model, 'exp': e, 'seed': s, 'n_iter': it,
                             **summary(v.QO[te], P, v.M[te], v.q95),
                             **peak_metrics(P, v.dates[te], obs, peaks, cfg['evaluation']['timing_window']),
                             **{k: sp[k] for k in S.SPEC if k in d.columns}})
            for h in range(H):
                pred_rows.append(pd.DataFrame({'model': model, 'exp': e, 'seed': s, 'ngay_goc': v.dates[te], 'h': h + 1,
                                               'Q_do': v.QO[te][:, h], 'dich_that': v.M[te][:, h], 'Q_du_bao': P[:, h]}))
        if verbose:
            m = pd.DataFrame(met_rows[-len(seeds):])
            print(f"{e:6s} (vòng {n_iter}): test NSE {m.NSE.mean():.3f} ± {m.NSE.std(ddof=0):.3f} | KGE {m.KGE.mean():.3f} "
                  f"| PBIAS {m.PBIAS.mean():+.1f} % | Peak Bias {m.Peak_Bias.mean():+.1f} %")
    if meta:                                           # một file mô tả cho mọi weight của mô hình
        (md_dir / 'mo_hinh.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    if logs:                                           # nhật ký huấn luyện theo epoch (mạng hồi tiếp)
        od = out_dir(cfg, model)
        od.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(logs).to_csv(od / f'train_log_{tag}{"_" + variant if variant else ""}.csv', index=False, encoding='utf-8-sig')
    met = pd.DataFrame(met_rows)
    S.write_metrics(cfg, met, model, H, 'test', S.nhom_of(variant))
    S.write_pred(cfg, pd.concat(pred_rows), model, H, 'test', S.nhom_of(variant))
    return met[[c for c in met.columns if c not in S.SPEC]]


def final_test(cfg, model, verbose=True, variant=None):
    cv = S.read_metrics(cfg, 'cv', 1, S.nhom_of(variant), model)
    return _final(cfg, model, cfg['horizons']['main'], FINAL_EXPS, cv, 'H1', verbose, variant)


# ---------------------------------------------------------------- thí nghiệm A: cửa sổ chung L = 3
def run_common_L(cfg, model, L=3, verbose=True):
    """Thí nghiệm A: mọi mô hình dùng cùng L. Mô hình nào ở thí nghiệm B (L chọn theo CV) đã chọn đúng L này thì hai
    thí nghiệm trùng nhau (quy trình tất định) → không chạy lại, ghi `same_as_base`. Ngược lại: dò lại các siêu tham số
    khác với L cố định rồi chạy đủ quy trình CV + test."""
    base, var = load_conf(cfg, model), f'L{L}'
    if model == 'persistence' or base['hp'].get('L') == L:
        save_conf(cfg, model, {**base, 'same_as_base': True}, var)
        return 'trùng thí nghiệm B'
    tune(cfg, model, verbose, var, L_fixed=L)
    cv_experiments(cfg, model, verbose, var)
    final_test(cfg, model, verbose, var)
    return 'chạy riêng'


# ---------------------------------------------------------------- 4. bài toán mở rộng 10 ngày
def extended(cfg, model, verbose=True):
    conf = load_conf(cfg, model)
    H = cfg['horizons']['extended']
    ds = get_dataset(cfg, H)
    seeds = seeds_for(cfg, model, 'cv')[:2]
    specs = _specs_base(model)[:2] if model == 'persistence' else [dict(exp='E1', fs='F0'), dict(exp='E2', fs='F1')]
    if model != 'persistence':
        specs.append(dict(exp='E5', fs='F1', gamma=conf['weights']['gamma'], cap=conf['weights']['cap']))
    rows = []
    for sp in specs:
        d = run_cv(cfg, ds, model, feature_list(cfg, sp['fs']), conf['hp'], seeds, sp.get('gamma', 0), sp.get('cap'))
        for k, x in sp.items():
            d[k] = x
        rows.append(d)
        if verbose:
            print(f"H=10 {sp['exp']}: NSE CV {cv_summary(d)['NSE CV']:.3f}")
    cv = pd.concat(rows, ignore_index=True)
    cv.insert(0, 'model', model)
    S.write_metrics(cfg, cv, model, H, 'cv')
    met = _final(cfg, model, H, [s['exp'] for s in specs], cv, 'H10', verbose)
    return cv_table(cv), met
