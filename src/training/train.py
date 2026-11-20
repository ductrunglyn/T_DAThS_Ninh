"""Huấn luyện / dự báo thống nhất cho mọi mô hình trên một View của bộ dữ liệu chuẩn.

fit_predict(cfg, v, model, hp, seed, ...) → (dự báo m³/s trên v.idx_eval [n, H], số vòng tốt nhất)
- Giai đoạn CV (n_iter = None): mô hình lặp (RNN/LSTM/GRU, XGBoost) dừng sớm theo tập val của fold.
- Giai đoạn cuối (n_iter cho trước): huấn luyện đúng n_iter epoch / cây, KHÔNG nhìn tập test.
"""
import copy
import random

import numpy as np
import torch
import torch.nn as nn

from src.models import rnn, lstm, gru
from src.models.persistence import Persistence
from src.models.ridge_arx import RidgeARX
from src.models.extra_trees import ExtraTrees
from src.models.xgboost_model import XGBoost
from src.training.losses import flow_weights, sample_weights, total_loss

DEEP = {'rnn': rnn, 'lstm': lstm, 'gru': gru}
FLAT = {'persistence': Persistence, 'ridge': RidgeARX, 'extra_trees': ExtraTrees, 'xgboost': XGBoost}
MODEL_NAMES = ['persistence', 'ridge', 'extra_trees', 'xgboost', 'rnn', 'lstm', 'gru']
DISPLAY = {'persistence': 'Persistence', 'ridge': 'Ridge-ARX', 'extra_trees': 'Extra Trees', 'xgboost': 'XGBoost',
           'rnn': 'RNN', 'lstm': 'LSTM', 'gru': 'GRU'}
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _t(a):
    return torch.as_tensor(a, dtype=torch.float32, device=DEV)


def train_deep(cfg, v, model_name, hp, seed, w_cells, lam=0.0, epochs=None, idx_eval=None):
    dc = cfg['deep']
    set_seed(seed)
    tr = v.idx_train
    ev = v.idx_eval if idx_eval is None else idx_eval
    use_fut = v.fut is not None
    net = DEEP[model_name].build(v.X_seq.shape[2], hp['hidden'], hp['layers'], dc['dropout'], v.Y.shape[1],
                                 v.fut.shape[1] if use_fut else 0).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=dc['lr'])
    Xtr, Ytr, Mtr, Wtr = _t(v.X_seq[tr]), _t(v.Y[tr]), _t(v.M[tr]), _t(w_cells[tr])
    Ftr = _t(v.fut[tr]) if use_fut else None
    if epochs is None:
        Xev, Yev, Mev, Wev = _t(v.X_seq[ev]), _t(v.Y[ev]), _t(v.M[ev]), _t(w_cells[ev])
        Fev = _t(v.fut[ev]) if use_fut else None
    g = torch.Generator().manual_seed(seed)
    best, best_state, best_ep, wait = np.inf, None, 0, 0
    n_ep = epochs if epochs is not None else dc['max_epochs']
    log = []                                   # nhật ký theo epoch (B3)
    for ep in range(n_ep):
        net.train()
        perm = torch.randperm(len(tr), generator=g).to(DEV)
        tl, gn, nonfinite = [], [], 0
        for i in range(0, len(tr), dc['batch']):
            b = perm[i:i + dc['batch']]
            loss = total_loss(net(Xtr[b], Ftr[b] if use_fut else None), Ytr[b], Mtr[b], Wtr[b], lam, dc['huber_delta'])
            opt.zero_grad(); loss.backward()
            g_norm = nn.utils.clip_grad_norm_(net.parameters(), 1.0)          # chuẩn gradient TRƯỚC khi cắt
            opt.step()
            tl.append(loss.item()); gn.append(float(g_norm))
            nonfinite += int(not np.isfinite(tl[-1]) or not np.isfinite(gn[-1]))
        with torch.no_grad():
            wn = float(torch.sqrt(sum((p_ ** 2).sum() for p_ in net.parameters())))
        rec = {'epoch': ep + 1, 'train_loss': float(np.mean(tl)), 'train_loss_max_batch': float(np.max(tl)),
               'grad_norm_mean': float(np.mean(gn)), 'grad_norm_max': float(np.max(gn)), 'weight_norm': wn,
               'nonfinite_batches': nonfinite}
        if epochs is not None:
            log.append(rec)
            continue
        net.eval()
        with torch.no_grad():
            vl = total_loss(net(Xev, Fev), Yev, Mev, Wev, lam, dc['huber_delta']).item()
        rec['val_loss'] = vl; log.append(rec)
        if vl < best - 1e-5:
            best, best_state, best_ep, wait = vl, copy.deepcopy(net.state_dict()), ep + 1, 0
        else:
            wait += 1
            if wait >= dc['patience']:
                break
    if epochs is None:
        net.load_state_dict(best_state)
    else:
        best_ep = epochs
    net.eval()
    with torch.no_grad():
        z = net(_t(v.X_seq[ev]), _t(v.fut[ev]) if use_fut else None).cpu().numpy()
    net.train_log = log
    return v.to_q(z), best_ep, net


def fit_predict(cfg, v, model_name, hp, seed=0, gamma=0, cap=None, lam=0.0, n_iter=None, idx_eval=None,
                return_model=False):
    ev = v.idx_eval if idx_eval is None else idx_eval
    w_cells = flow_weights(v.W_q, v.M, v.idx_train, v.q95, gamma, cap)
    if model_name in DEEP:
        P, it, net = train_deep(cfg, v, model_name, hp, seed, w_cells, lam, n_iter, ev)
        return (P, it, net) if return_model else (P, it)
    assert not lam, 'phạt dự báo thiếu chỉ dùng cho mô hình học sâu'
    # mô hình dạng bảng: không có mặt nạ từng ô → dùng mẫu train có đủ mọi ô đích thật
    tr = v.idx_train[v.M[v.idx_train].all(1)]
    sw = sample_weights(w_cells[tr], v.M[tr]) if gamma else None
    kw = dict(cfg['models'].get(model_name, {}).get('fixed', {}))
    kw.update({k: x for k, x in hp.items() if k != 'L'})
    if model_name == 'xgboost':
        kw['seed'] = seed
        if n_iter is not None:
            kw['n_estimators'] = int(n_iter)
    if model_name == 'extra_trees':
        kw['seed'] = seed
    m = FLAT[model_name](**kw) if model_name != 'persistence' else Persistence()
    eval_set = None
    if model_name == 'xgboost' and n_iter is None:
        e2 = ev[v.M[ev].all(1)]
        eval_set = (e2, v.Y[e2], sample_weights(w_cells[e2], v.M[e2]) if gamma else None)
    m.fit(v, tr, v.Y[tr], sw, eval_set)
    P = m.predict(v, ev)
    it = getattr(m, 'best_iter', None) or 0
    return (P, it, m) if return_model else (P, it)
