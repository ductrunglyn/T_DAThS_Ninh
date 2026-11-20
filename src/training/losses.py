"""Hàm mất mát và trọng số mẫu dùng chung."""
import numpy as np
import torch
import torch.nn.functional as F


def flow_weights(W_q, M, idx_train, q95, gamma=0, cap=None):
    """w = (1 + Q/Q95_train)^γ trên từng ô đích, chặn trên `cap` (nếu có), chuẩn hoá trung bình = 1 trên các ô
    đích thật của train. γ = 0 → w = 1. Q95 lấy từ train của chính fold."""
    if not gamma:
        return np.ones_like(W_q, dtype=np.float32)
    w = (1.0 + W_q / q95) ** gamma
    if cap:
        w = np.minimum(w, cap)
    m = M[idx_train]
    w = w / w[idx_train][m].mean()
    return w.astype(np.float32)


def sample_weights(w_cells, M):
    """Trọng số một mẫu cho mô hình dạng bảng (không có mặt nạ từng ô): trung bình trên các ô đích thật."""
    return (w_cells * M).sum(1) / np.maximum(M.sum(1), 1)


def masked_huber(p, y, m, w, delta=1.0):
    l = F.huber_loss(p, y, reduction='none', delta=delta)
    mw = m * w
    return (l * mw).sum() / mw.sum()


def underprediction(p, y, m):
    return ((F.relu(y - p) ** 2) * m).sum() / m.sum()


def total_loss(p, y, m, w, lam=0.0, delta=1.0):
    l = masked_huber(p, y, m, w, delta)
    if lam:
        l = l + lam * underprediction(p, y, m)
    return l
