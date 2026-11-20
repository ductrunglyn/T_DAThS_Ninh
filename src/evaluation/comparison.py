"""Tổng hợp kết quả của mọi mô hình (đọc file đã lưu) và so sánh thống kê trên cùng ngày đích."""
import numpy as np
import pandas as pd

from src import results_store as S
from src.training.train import MODEL_NAMES, DISPLAY
from src.training.tuning import cv_summary
from src.evaluation.bootstrap import block_indices, compare, compare_multi
from src.evaluation.metrics import DIGITS, LABEL


def load_all(cfg, tag='H1', variant=None):
    """Đọc kết quả mọi mô hình từ các file gộp (results/ket_qua_thuc_nghiem.csv, results/du_bao.csv.gz): chỉ tiêu CV,
    chỉ tiêu test, dự báo test. variant='L3' (thí nghiệm A): mô hình có `same_as_base` dùng kết quả thí nghiệm B."""
    H = int(tag[1:])
    cv, met, pred = [], [], []
    for m in MODEL_NAMES:
        nhom = 'chinh'
        if variant:
            c = S.load_conf(cfg, m, S.nhom_of(variant))
            if c and not c.get('same_as_base'):
                nhom = S.nhom_of(variant)
        c, t = S.read_metrics(cfg, 'cv', H, nhom, m), S.read_metrics(cfg, 'test', H, nhom, m)
        if len(c):
            cv.append(c)
        if len(t):
            met.append(t)
            pred.append(S.read_pred(cfg, 'test', H, nhom, m))
    return pd.concat(cv), pd.concat(met), pd.concat(pred)


def cv_matrix(cv, value='NSE CV'):
    rows = []
    for (m, e), d in cv.groupby(['model', 'exp']):
        rows.append({'mô hình': DISPLAY[m], 'thực nghiệm': e, value: cv_summary(d)[value]})
    t = pd.DataFrame(rows).pivot(index='mô hình', columns='thực nghiệm', values=value)
    return t.reindex([DISPLAY[m] for m in MODEL_NAMES if DISPLAY[m] in t.index])


def fmt(v, nd):
    """TB ± ĐLC giữa các hạt giống. ĐLC mẫu (ddof = 1), thống nhất với bảng độ ổn định hạt giống và bảng theo fold."""
    v = np.asarray(v, float)
    return f'{v.mean():.{nd}f} ± {v.std(ddof=1):.{nd}f}' if len(v) > 1 else f'{v.mean():.{nd}f}'


def test_table(met, exps, metrics):
    rows = []
    for m in MODEL_NAMES:
        for e in exps:
            d = met[(met.model == m) & (met.exp == e)]
            if len(d):
                rows.append({'mô hình': DISPLAY[m], 'thực nghiệm': e,
                             **{LABEL[k]: fmt(d[k], DIGITS[k]) for k in metrics}})
    return pd.DataFrame(rows).set_index(['mô hình', 'thực nghiệm'])


def arrays(pred, model, exp):
    d = pred[(pred.model == model) & (pred.exp == exp)]
    if not len(d):
        return None
    seeds = sorted(d.seed.unique())
    O = d[d.seed == seeds[0]].pivot(index='ngay_goc', columns='h', values='Q_do')
    M = d[d.seed == seeds[0]].pivot(index='ngay_goc', columns='h', values='dich_that').astype(bool)
    P = [d[d.seed == s].pivot(index='ngay_goc', columns='h', values='Q_du_bao').loc[O.index].values for s in seeds]
    return O.values, M.values, P, O.index


def bootstrap_table(cfg, pred, pairs):
    """pairs: [(model_a, exp_a, model_b, exp_b, mô tả)] — dùng CHUNG một bộ chỉ số bootstrap."""
    bc = cfg['bootstrap']
    rows, B, ref = [], None, None
    for ma, ea, mb, eb, desc in pairs:
        A, Bm = arrays(pred, ma, ea), arrays(pred, mb, eb)
        if A is None or Bm is None:
            continue
        assert A[3].equals(Bm[3]) and np.allclose(A[0], Bm[0]), 'hai mô hình phải dự báo cùng ngày đích'
        if B is None:
            B, ref = block_indices(len(A[0]), bc['block'], bc['n'], bc['seed']), A[3]
        assert A[3].equals(ref)
        d0, lo, hi, sig = compare(A[0], A[1], A[2], Bm[2], B)
        rows.append({'so sánh': desc, 'A': f'{DISPLAY[ma]} {ea}', 'B': f'{DISPLAY[mb]} {eb}',
                     'ΔNSE (A − B)': round(d0, 3), 'KTC 95 %': f'[{lo:+.3f}; {hi:+.3f}]', 'có ý nghĩa': '✔' if sig else ''})
    return pd.DataFrame(rows)


def bootstrap_multi(cfg, pred, pairs):
    """Như bootstrap_table nhưng cho NSE, KGE, MAE, RMSE và tỉ lệ bootstrap A tốt hơn B."""
    bc = cfg['bootstrap']
    rows, B, ref = [], None, None
    for ma, ea, mb, eb, desc in pairs:
        A, Bm = arrays(pred, ma, ea), arrays(pred, mb, eb)
        if A is None or Bm is None:
            continue
        assert A[3].equals(Bm[3]) and np.allclose(A[0], Bm[0])
        if B is None:
            B, ref = block_indices(len(A[0]), bc['block'], bc['n'], bc['seed']), A[3]
        assert A[3].equals(ref)
        r = compare_multi(A[0], A[1], A[2], Bm[2], B)
        row = {'so sánh': desc}
        for k, x in r.items():
            nd = 3 if k in ('NSE', 'KGE') else 1
            row[f'Δ{k}'] = round(x['delta'], nd)
            row[f'KTC {k}'] = f"[{x['lo']:+.{nd}f}; {x['hi']:+.{nd}f}]" + (' ✔' if x['sig'] else '')
            row[f'P(A tốt hơn) {k}'] = round(x['p_better'], 3)
        rows.append(row)
    return pd.DataFrame(rows).set_index('so sánh')


def fold_table(cv, exps=None):
    """Chỉ tiêu CV theo từng fold (trung bình hạt giống) + TB ± ĐLC giữa fold, thời gian, số vòng."""
    rows = []
    for (m, e), d in cv.groupby(['model', 'exp'], sort=False):
        if exps and e not in exps:
            continue
        g = d.groupby('fold').mean(numeric_only=True)
        row = {'mô hình': DISPLAY[m], 'thực nghiệm': e}
        for k in ['NSE', 'KGE', 'MAE', 'RMSE', 'PBIAS']:
            for f in g.index:
                row[f'{k} {f}'] = g.loc[f, k]
            row[f'{k} TB'] = g[k].mean(); row[f'{k} ĐLC'] = g[k].std(ddof=1)
        row['thời gian/lần (s)'] = d['time_s'].mean() if 'time_s' in d else np.nan
        row['vòng (trung vị)'] = d['best_iter'].median()
        rows.append(row)
    return pd.DataFrame(rows).set_index(['mô hình', 'thực nghiệm'])


def one_se(tab, simplicity):
    """Quy tắc một sai số chuẩn (phân tích phụ, khai báo trước): trong các cấu hình có NSE CV ≥ max − SE (SE = ĐLC giữa
    3 fold / √3 của cấu hình tốt nhất), chọn cấu hình 'đơn giản nhất' theo khoá `simplicity` (tăng dần)."""
    best = tab.iloc[0]
    se = best['NSE CV (ĐLC giữa fold)'] / np.sqrt(3)
    ok = tab[tab['NSE CV'] >= best['NSE CV'] - se]
    return ok.sort_values(simplicity).iloc[0], se


def peak_table(pred, peaks, models_exps, window=3):
    """Mô tả từng đỉnh: Q đo, dự báo t+1 tại ngày đỉnh (TB hạt giống), cực đại dự báo trong ±window ngày và độ lệch thời gian."""
    rows = []
    for m, e in models_exps:
        r = arrays(pred, m, e)
        if r is None:
            continue
        O, M, P, idx = r
        s = pd.Series(np.mean(P, axis=0)[:, 0], index=idx + pd.Timedelta(days=1))
        o = pd.Series(O[:, 0], index=idx + pd.Timedelta(days=1))
        for d in peaks:
            w = s.loc[d - pd.Timedelta(days=window):d + pd.Timedelta(days=window)]
            rows.append({'đỉnh': d.date(), 'Q đo': o[d], 'mô hình': f'{DISPLAY[m]} {e}', 'dự báo tại ngày đỉnh': s[d],
                         'sai số tại đỉnh (%)': 100 * (s[d] - o[d]) / o[d], 'cực đại dự báo ±3 ngày': w.max(),
                         'ngày cực đại dự báo − ngày đỉnh': (w.idxmax() - d).days,
                         'thiếu so với đỉnh (%)': 100 * (w.max() - o[d]) / o[d]})
    return pd.DataFrame(rows)


def sensitivity_excluding(pred, flags, models_exps):
    """Độ nhạy: NSE khi BỎ các ngày đích bị cờ (chỉ khi chấm điểm; nhãn và huấn luyện giữ nguyên)."""
    from src.evaluation.metrics import summary
    fl = set(pd.to_datetime(flags))
    rows = []
    for m, e in models_exps:
        r = arrays(pred, m, e)
        if r is None:
            continue
        O, M, P, idx = r
        tgt = np.array([[d + pd.Timedelta(days=h + 1) in fl for h in range(O.shape[1])] for d in idx])
        M2 = M & ~tgt
        rows.append({'mô hình': f'{DISPLAY[m]} {e}', 'số ô đích bị cờ': int((M & tgt).sum()),
                     'NSE đầy đủ': np.mean([summary(O, p, M)['NSE'] for p in P]),
                     'NSE bỏ ngày bị cờ': np.mean([summary(O, p, M2)['NSE'] for p in P])})
    t = pd.DataFrame(rows)
    t['Δ'] = t['NSE bỏ ngày bị cờ'] - t['NSE đầy đủ']
    return t
