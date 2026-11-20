"""Tập mẫu chung và vai trò của mỗi mẫu trong từng fold (rolling-origin) và trong giai đoạn cuối.

Một mẫu = một ngày gốc t: đầu vào [t−L+1 … t], đích Q_CĐ[t+1 … t+H].
- Mẫu bị loại (`reliable = False`) nếu cửa sổ max_window ngày [t−max_window+1 … t] chạm ngày thuộc khoảng thiếu dài
  (cùng một tập mẫu cho mọi L để mọi cấu hình được so sánh trên cùng ngày đích).
- Fold k: train nếu t+H ≤ train_end; val nếu t ≥ val_start và t+H ≤ val_end. Đầu vào của mẫu val được phép dùng
  quan trắc thuộc giai đoạn train (đúng như khi dự báo thực tế); ngày ĐÍCH của train và val không trùng nhau.
- Giai đoạn cuối: train nếu t+H ≤ 31/12/2024; test nếu t ≥ 01/01/2025.
"""
import numpy as np
import pandas as pd


def make_samples(index, H, first_valid, max_window, unreliable, target_real, cfg):
    n = len(index)
    t0 = first_valid + max_window - 1
    pos = np.arange(t0, n - H)
    dates = index[pos]
    unrel = np.asarray(unreliable, bool)
    reliable = np.array([not unrel[t - max_window + 1:t + 1].any() for t in pos])
    real = np.asarray(target_real, bool)
    M = np.stack([real[t + 1:t + H + 1] for t in pos])
    S = pd.DataFrame({'pos': pos, 'date': dates, 'reliable': reliable, 'n_real': M.sum(1)})
    DH = pd.Timedelta(days=H)
    for fd in cfg['splits']['folds']:
        role = np.full(len(pos), '', object)
        role[(dates + DH <= fd['train_end']) & reliable] = 'train'
        role[(dates >= fd['val_start']) & (dates + DH <= fd['val_end']) & reliable] = 'val'
        S[fd['name']] = role
    fn = cfg['splits']['final']
    role = np.full(len(pos), '', object)
    role[(dates + DH <= fn['train_end']) & reliable] = 'train'
    role[dates >= fn['test_start']] = 'test'
    S[fn['name']] = role
    assert S.loc[S[fn['name']] == 'test', 'reliable'].all(), 'mẫu test không được chạm khoảng thiếu dài'
    return S, M


def split_info(cfg, name):
    """(train_end, vai trò đánh giá) của một fold hoặc FINAL."""
    if name == cfg['splits']['final']['name']:
        return cfg['splits']['final']['train_end'], 'test'
    fd = next(f for f in cfg['splits']['folds'] if f['name'] == name)
    return fd['train_end'], 'val'


def fold_names(cfg):
    return [f['name'] for f in cfg['splits']['folds']]
