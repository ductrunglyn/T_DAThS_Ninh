"""Bootstrap khối dịch chuyển trên các ngày gốc test — DÙNG CHUNG một bộ chỉ số cho mọi so sánh."""
import numpy as np


def block_indices(n, block=30, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    B = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    return B


def nse_boot(O, M, P_list, B):
    """NSE (trung bình theo h, rồi trung bình theo hạt giống) trên từng mẫu bootstrap. Trả về mảng [n_boot]."""
    O0 = np.nan_to_num(O)
    Ob, Mb = O0[B], M[B].astype(float)                 # [n_boot, n, H]
    msum = Mb.sum(1)
    mean = (Ob * Mb).sum(1) / msum
    sst = (((Ob - mean[:, None, :]) ** 2) * Mb).sum(1)
    out = []
    for P in P_list:
        sse = (((Ob - P[B]) ** 2) * Mb).sum(1)
        out.append((1 - sse / sst).mean(1))
    return np.mean(out, axis=0)


def nse_point(O, M, P_list):
    B = np.arange(len(O))[None, :]
    return float(nse_boot(O, M, P_list, B)[0])


def compare(O, M, PA, PB, B):
    """ΔNSE = A − B, khoảng tin cậy 95 % (bootstrap khối, cùng chỉ số B cho A và B)."""
    d0 = nse_point(O, M, PA) - nse_point(O, M, PB)
    db = nse_boot(O, M, PA, B) - nse_boot(O, M, PB, B)
    lo, hi = np.percentile(db, [2.5, 97.5])
    return d0, lo, hi, bool(lo > 0 or hi < 0)


# ---------------------------------------------------------------- nhiều chỉ tiêu (B8)
def metrics_boot(O, M, P_list, B):
    """NSE, KGE, MAE, RMSE (trung bình theo h, rồi theo hạt giống) trên từng mẫu bootstrap → dict {tên: [n_boot]}."""
    O0 = np.nan_to_num(O)
    Ob, Mb = O0[B], M[B].astype(float)
    n = Mb.sum(1)
    mo = (Ob * Mb).sum(1) / n
    so = np.sqrt((((Ob - mo[:, None, :]) ** 2) * Mb).sum(1) / n)
    sst = (((Ob - mo[:, None, :]) ** 2) * Mb).sum(1)
    acc = {k: [] for k in ('NSE', 'KGE', 'MAE', 'RMSE')}
    for P in P_list:
        Pb = P[B]
        e = (Pb - Ob) * Mb
        mp = (Pb * Mb).sum(1) / n
        sp = np.sqrt((((Pb - mp[:, None, :]) ** 2) * Mb).sum(1) / n)
        cov = (((Ob - mo[:, None, :]) * (Pb - mp[:, None, :])) * Mb).sum(1) / n
        r = cov / (so * sp)
        acc['NSE'].append((1 - (e ** 2).sum(1) / sst).mean(1))
        acc['KGE'].append((1 - np.sqrt((r - 1) ** 2 + (sp / so - 1) ** 2 + (mp / mo - 1) ** 2)).mean(1))
        acc['MAE'].append((np.abs(e).sum(1) / n).mean(1))
        acc['RMSE'].append(np.sqrt((e ** 2).sum(1) / n).mean(1))
    return {k: np.mean(v, axis=0) for k, v in acc.items()}


HIGHER_BETTER = {'NSE': True, 'KGE': True, 'MAE': False, 'RMSE': False}


def compare_multi(O, M, PA, PB, B):
    """Với mỗi chỉ tiêu: Δ = A − B trên toàn tập, KTC 95 % bootstrap, tỉ lệ lần bootstrap A tốt hơn B."""
    full = np.arange(len(O))[None, :]
    a0, b0 = metrics_boot(O, M, PA, full), metrics_boot(O, M, PB, full)
    ab, bb = metrics_boot(O, M, PA, B), metrics_boot(O, M, PB, B)
    out = {}
    for k in a0:
        d = ab[k] - bb[k]
        lo, hi = np.percentile(d, [2.5, 97.5])
        better = (d > 0) if HIGHER_BETTER[k] else (d < 0)
        out[k] = {'delta': float(a0[k][0] - b0[k][0]), 'lo': float(lo), 'hi': float(hi), 'p_better': float(better.mean()),
                  'sig': bool(lo > 0 or hi < 0)}
    return out
