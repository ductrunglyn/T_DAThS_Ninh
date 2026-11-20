"""Tạo đặc trưng từ bảng đã điền nhân quả. Mọi đặc trưng tại ngày t chỉ dùng dữ liệu ≤ t.
Giá trị ở đơn vị vật lý; biến đổi (log, chuẩn hoá) nằm ở preprocessing.py."""
import numpy as np
import pandas as pd

from src.data.quality_control import repeat_length_causal

# tên → (đơn vị, biến đổi, nhóm, mô tả)
FEATURE_META = {
    'q_cd':        ('m³/s', 'log1p', 'Cửa Đạt', 'Lưu lượng đến hồ Cửa Đạt (đích và lịch sử)'),
    'qcd_ema3':    ('m³/s', 'log1p', 'Cửa Đạt', 'Trung bình trượt mũ nhân quả (span 3) của Q đến CĐ'),
    'mua_v1':      ('mm', 'log1p', 'Mưa V1', 'Mưa ngày Vùng 1 (lưu vực Hủa Na)'),
    'mua_v1_3':    ('mm', 'log1p', 'Mưa V1', 'Tổng mưa 3 ngày t−2…t'),
    'mua_v1_7':    ('mm', 'log1p', 'Mưa V1', 'Tổng mưa 7 ngày'),
    'mua_v1_15':   ('mm', 'log1p', 'Mưa V1', 'Tổng mưa 15 ngày'),
    'mua_v2':      ('mm', 'log1p', 'Mưa V2', 'Mưa ngày Vùng 2 (khu giữa Hủa Na – Cửa Đạt)'),
    'mua_v2_3':    ('mm', 'log1p', 'Mưa V2', 'Tổng mưa 3 ngày'),
    'mua_v2_7':    ('mm', 'log1p', 'Mưa V2', 'Tổng mưa 7 ngày'),
    'mua_v2_15':   ('mm', 'log1p', 'Mưa V2', 'Tổng mưa 15 ngày'),
    'doy_sin':     ('—', 'none', 'Lịch', 'sin(2π·ngày trong năm/365,25)'),
    'doy_cos':     ('—', 'none', 'Lịch', 'cos(2π·ngày trong năm/365,25)'),
    'hn_qxa':      ('m³/s', 'log1p', 'Hủa Na', 'Q ra Hủa Na = Q nhà máy + Q tràn'),
    'hn_qden':     ('m³/s', 'log1p', 'Hủa Na', 'Q đến Hủa Na'),
    'hn_mn':       ('m', 'none', 'Hủa Na', 'Mực nước hồ Hủa Na Z'),
    'hn_qnm':      ('m³/s', 'log1p', 'Hủa Na', 'Q qua nhà máy Hủa Na'),
    'hn_qtran':    ('m³/s', 'log1p', 'Hủa Na', 'Q qua tràn Hủa Na'),
    'hn_gio':      ('giờ', 'none', 'Hủa Na', 'Số giờ phát điện Hủa Na'),
    'hn_dmn':      ('m', 'none', 'Hủa Na (thủy văn)', 'ΔZ = Z(t) − Z(t−1)'),
    'hn_bal':      ('m³/s', 'slog', 'Hủa Na (thủy văn)', 'Q đến − Q ra (log có dấu)'),
    'hn_ratio':    ('—', 'log1p', 'Hủa Na (thủy văn)', 'Q ra / (Q đến + 1), cắt ở 10'),
    'hn_spill':    ('0/1', 'binary', 'Hủa Na (thủy văn)', 'Q tràn > ngưỡng xả lũ'),
    'cd_miss':     ('0/1', 'binary', 'Chất lượng', 'Ngày thiếu Q CĐ (giá trị đang được giữ từ trước)'),
    'cd_age':      ('ngày', 'log1p', 'Chất lượng', 'Số ngày kể từ quan trắc Q CĐ gần nhất'),
    'hn_miss':     ('0/1', 'binary', 'Chất lượng', 'Ngày thiếu số liệu Hủa Na'),
    'hn_age':      ('ngày', 'log1p', 'Chất lượng', 'Số ngày kể từ quan trắc Hủa Na gần nhất'),
    'qhn_rep_flag': ('0/1', 'binary', 'Chất lượng', 'Q đến HN đã không đổi ≥ repeat_min ngày tính đến t'),
    'qhn_rep_len': ('ngày', 'log1p', 'Chất lượng', 'Số ngày liên tiếp Q đến HN bằng giá trị ngày trước (tính đến t)'),
}


def build_features(filled, cfg):
    f = pd.DataFrame(index=filled.index)
    f['q_cd'] = filled['q_cd']
    f['qcd_ema3'] = filled['q_cd'].ewm(span=3, adjust=False).mean()
    for v in ['v1', 'v2']:
        f[f'mua_{v}'] = filled[f'mua_{v}']
        for k in (3, 7, 15):
            f[f'mua_{v}_{k}'] = filled[f'mua_{v}'].rolling(k, min_periods=k).sum()
    doy = filled.index.dayofyear
    f['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    f['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
    for c in ['hn_qxa', 'hn_qden', 'hn_mn', 'hn_qnm', 'hn_qtran', 'hn_gio']:
        f[c] = filled[c]
    f['hn_dmn'] = filled['hn_mn'].diff()
    f['hn_bal'] = filled['hn_qden'] - filled['hn_qxa']
    f['hn_ratio'] = np.clip(filled['hn_qxa'] / (filled['hn_qden'] + 1.0), 0, 10)
    f['hn_spill'] = (filled['hn_qtran'] > cfg['quality']['spill_threshold']).astype(float)
    for c in ['cd_miss', 'cd_age', 'hn_miss', 'hn_age']:
        f[c] = filled[c]
    # giá trị lặp: tính trên chuỗi QUAN TRẮC (ngày thiếu ngắt chuỗi lặp)
    obs_qden = filled['hn_qden'].where(~filled['hn_missing'])
    rl = repeat_length_causal(obs_qden.values)
    f['qhn_rep_len'] = rl
    f['qhn_rep_flag'] = (rl >= cfg['quality']['repeat_min'] - 1).astype(float)
    assert list(f.columns) == [c for c in f.columns if c in FEATURE_META]
    return f


def feature_list(cfg, set_name):
    groups = cfg['feature_groups']
    out = []
    for item in cfg['feature_sets'][set_name]:
        out += groups[item] if item in groups else [item]
    assert len(out) == len(set(out)) and all(c in FEATURE_META for c in out), set_name
    return out


def metadata_table():
    return pd.DataFrame([{'biến': k, 'đơn vị': u, 'biến đổi': t, 'nhóm': g, 'mô tả': d}
                         for k, (u, t, g, d) in FEATURE_META.items()]).set_index('biến')
