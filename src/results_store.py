"""Lưu trữ kết quả thực nghiệm GỘP — mọi mô hình × kịch bản × hạt giống trong ít file:

    results/ket_qua_thuc_nghiem.csv   chỉ tiêu CV (theo fold × hạt giống) và test (theo hạt giống), dạng dài
    results/du_bao.csv.gz             dự báo theo ngày (CV và test) — để tính bootstrap, vẽ hình, kiểm thử không cần huấn luyện lại
    results/do_sieu_tham_so.csv       lưới dò siêu tham số của mọi mô hình
    results/cau_hinh_chon.json        cấu hình chọn theo CV (siêu tham số, γ của E5, λ của E6) của mọi mô hình

Cột phân nhóm chung:
    nhom       'chinh' (thí nghiệm B, kết quả chính) · 'A_L3' (thí nghiệm A, cửa sổ chung L = 3) · 'KH' (kịch bản kế hoạch xả)
    H          1 (bài toán t+1) · 10 (t+1 … t+10)
    giai_doan  'cv' (tập xác thực của fold F1, F2, F3) · 'test' (tập kiểm tra, fold = 'FINAL')

Ghi theo kiểu thay thế: mỗi lần ghi thay toàn bộ các dòng cùng (nhom, H, giai_doan, model) — chạy lại một mô hình không làm
trùng dòng. Hàm đọc trả về bảng đúng định dạng các file cũ theo từng mô hình (cv: best_iter; test: n_iter).
"""
import json

import numpy as np
import pandas as pd

from src.utils import path

GROUP = ['nhom', 'H', 'giai_doan', 'model']
METRICS = ['MSE', 'MAE', 'RMSE', 'r2', 'NSE', 'KGE', 'PBIAS', 'UNDER', 'NSE_Q95', 'RMSE_Q95', 'UNDER_Q95',
           'Peak_RMSE', 'Peak_Bias', 'Peak_Timing']
SPEC = ['fs', 'gamma', 'cap', 'lam', 'fut']
MET_COLS = GROUP + ['exp', 'fold', 'seed', 'so_vong', 'time_s'] + METRICS + SPEC
PRED_COLS = GROUP + ['exp', 'fold', 'seed', 'ngay_goc', 'h', 'Q_do', 'dich_that', 'Q_du_bao']
INT_HP = ['L', 'hidden', 'layers', 'min_samples_leaf', 'max_depth', 'min_child_weight']
def nhom_of(variant):
    """variant của pipeline → nhóm: None → 'chinh'; 'L3' (thí nghiệm A, L chung) → 'A_L3'; 'KH' → 'KH'."""
    if variant is None:
        return 'chinh'
    return f'A_{variant}' if variant.startswith('L') else variant


def files(cfg):
    r = path(cfg, 'results')
    return {'met': r / 'ket_qua_thuc_nghiem.csv', 'pred': r / 'du_bao.csv.gz',
            'tune': r / 'do_sieu_tham_so.csv', 'conf': r / 'cau_hinh_chon.json'}


# ---------------------------------------------------------------------------------------------------------------- nội bộ
ORDER = {'nhom': ['chinh', 'A_L3', 'KH'], 'giai_doan': ['cv', 'test'],
         'model': ['persistence', 'ridge', 'extra_trees', 'xgboost', 'rnn', 'lstm', 'gru']}
_CACHE = {}


def _read(f, cols):
    """Đọc có bộ nhớ đệm theo thời điểm sửa file (file dự báo ~1,2 triệu dòng chỉ đọc một lần mỗi phiên)."""
    if not f.exists():
        return pd.DataFrame(columns=cols)
    key = (str(f), f.stat().st_mtime_ns)
    if key not in _CACHE:
        for k in [k for k in _CACHE if k[0] == str(f)]:
            del _CACHE[k]
        _CACHE[key] = pd.read_csv(f, encoding='utf-8-sig', dtype={'fold': str}, float_precision='round_trip')
    return _CACHE[key].copy()


def _eq(s, v):
    if pd.api.types.is_numeric_dtype(s):
        return s == float(v)
    return s.astype(str) == str(v)


def _replace(f, new, keys, cols, **wkw):
    """Thay mọi dòng cũ trùng `keys` (dict cột → giá trị) bằng `new`, giữ nguyên các dòng khác. Sắp theo nhóm, mô hình
    (thứ tự cố định); trong một nhóm giữ nguyên thứ tự dòng lúc ghi."""
    old = _read(f, cols)
    if len(old):
        m = np.ones(len(old), bool)
        for k, v in keys.items():
            m &= _eq(old[k], v).values
        old = old[~m]
    out = pd.concat([d for d in (old, new.reindex(columns=cols)) if len(d)], ignore_index=True)
    rank = {k: out[k].map({v: i for i, v in enumerate(o)}).fillna(len(o)) for k, o in ORDER.items()}
    out = out.assign(**{f'_{k}': r for k, r in rank.items()}) \
        .sort_values(['_nhom', 'H', '_giai_doan', '_model'], kind='stable')[cols]
    f.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(f, index=False, encoding='utf-8-sig', **wkw)


def _filter(d, **kw):
    for k, v in kw.items():
        if v is None:
            continue
        d = d[d[k].isin(v)] if isinstance(v, (list, tuple, set)) else d[_eq(d[k], v)]
    return d


# ---------------------------------------------------------------------------------------------------------------- chỉ tiêu
def write_metrics(cfg, df, model, H, giai_doan, nhom='chinh'):
    d = df.copy()
    d = d.rename(columns={'best_iter': 'so_vong', 'n_iter': 'so_vong'})
    d['nhom'], d['H'], d['giai_doan'], d['model'] = nhom, H, giai_doan, model
    if giai_doan == 'test':
        d['fold'] = 'FINAL'
    _replace(files(cfg)['met'], d, dict(nhom=nhom, H=H, giai_doan=giai_doan, model=model), MET_COLS)


def read_metrics(cfg, giai_doan, H=1, nhom='chinh', model=None, exp=None):
    """Bảng chỉ tiêu theo định dạng cũ: cv → model, fold, seed, best_iter, time_s, chỉ tiêu, exp, fs, gamma, cap, lam, fut;
    test → model, exp, seed, n_iter, chỉ tiêu. Giữ thứ tự dòng như lúc ghi."""
    d = _read(files(cfg)['met'], MET_COLS)
    d = _filter(d, nhom=nhom, H=H, giai_doan=giai_doan, model=model, exp=exp)
    if giai_doan == 'cv':
        d = d.rename(columns={'so_vong': 'best_iter'})
        cols = ['model', 'fold', 'seed', 'best_iter', 'time_s'] + METRICS[:11] + ['exp'] + SPEC
    else:
        d = d.rename(columns={'so_vong': 'n_iter'})
        cols = ['model', 'exp', 'seed', 'n_iter'] + METRICS
    d = d[cols].reset_index(drop=True)
    return d.dropna(axis=1, how='all') if giai_doan == 'cv' else d


def models_with(cfg, giai_doan='test', H=1, nhom='chinh'):
    d = _read(files(cfg)['met'], MET_COLS)
    return list(dict.fromkeys(_filter(d, nhom=nhom, H=H, giai_doan=giai_doan)['model']))


# ---------------------------------------------------------------------------------------------------------------- dự báo
def write_pred(cfg, df, model, H, giai_doan, nhom='chinh'):
    d = df.copy()
    d['nhom'], d['H'], d['giai_doan'], d['model'] = nhom, H, giai_doan, model
    if giai_doan == 'test':
        d['fold'] = 'FINAL'
    d['ngay_goc'] = pd.to_datetime(d['ngay_goc']).dt.strftime('%Y-%m-%d')
    _replace(files(cfg)['pred'], d, dict(nhom=nhom, H=H, giai_doan=giai_doan, model=model), PRED_COLS,
             compression={'method': 'gzip', 'compresslevel': 6, 'mtime': 0})


def read_pred(cfg, giai_doan, H=1, nhom='chinh', model=None, exp=None):
    """Dự báo theo định dạng cũ: test → model, exp, seed, ngay_goc, h, Q_do, dich_that, Q_du_bao;
    cv → thêm cột fold. ngay_goc là datetime, dich_that là bool."""
    d = _read(files(cfg)['pred'], PRED_COLS)
    d = _filter(d, nhom=nhom, H=H, giai_doan=giai_doan, model=model, exp=exp)
    cols = ['model', 'exp'] + (['fold'] if giai_doan == 'cv' else []) + ['seed', 'ngay_goc', 'h', 'Q_do', 'dich_that', 'Q_du_bao']
    d = d[cols].reset_index(drop=True)
    d['ngay_goc'] = pd.to_datetime(d['ngay_goc'])
    d['dich_that'] = d['dich_that'].astype(str).eq('True')
    return d


# ---------------------------------------------------------------------------------------------------------------- dò siêu tham số
def write_tuning(cfg, tab, model, nhom='chinh'):
    d = tab.copy()
    d.insert(0, 'model', model)
    d.insert(0, 'nhom', nhom)
    f = files(cfg)['tune']
    old = _read(f, ['nhom', 'model'])
    if len(old):
        old = old[~((old.nhom == nhom) & (old.model == model))]
    out = pd.concat([x for x in (old, d) if len(x)], ignore_index=True)
    hp = [c for c in out.columns if c not in ('nhom', 'model') and not c.startswith(('NSE', 'KGE', 'r2', 'PBIAS', 'vòng'))]
    rest = [c for c in out.columns if c not in hp and c not in ('nhom', 'model')]
    out[['nhom', 'model'] + hp + rest].to_csv(f, index=False, encoding='utf-8-sig')


def read_tuning(cfg, model, nhom='chinh'):
    d = _read(files(cfg)['tune'], ['nhom', 'model'])
    d = d[(d.nhom == nhom) & (d.model == model)].drop(columns=['nhom', 'model']).dropna(axis=1, how='all')
    for c in INT_HP:
        if c in d and (d[c].dropna() % 1 == 0).all():
            d[c] = d[c].astype(int)
    return d.reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------------------- cấu hình chọn
def _conf_all(cfg):
    f = files(cfg)['conf']
    return json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}


def load_conf(cfg, model, nhom='chinh'):
    return dict(_conf_all(cfg).get(nhom, {}).get(model, {}))


def save_conf(cfg, model, conf, nhom='chinh'):
    allc = _conf_all(cfg)
    allc.setdefault(nhom, {})[model] = conf
    order = ['persistence', 'ridge', 'extra_trees', 'xgboost', 'rnn', 'lstm', 'gru']
    allc = {g: {m: allc[g][m] for m in sorted(allc[g], key=lambda x: order.index(x) if x in order else 99)}
            for g in sorted(allc, key=lambda g: ['chinh', 'A_L3'].index(g) if g in ('chinh', 'A_L3') else 9)}
    files(cfg)['conf'].write_text(json.dumps(allc, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
