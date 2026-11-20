"""Điền thiếu NHÂN QUẢ theo độ dài khoảng thiếu.

- Khoảng thiếu ≤ max_gap ngày: giữ giá trị quan trắc gần nhất trước đó (forward fill) + cờ thiếu + tuổi số liệu.
- Khoảng thiếu > max_gap ngày: vẫn forward fill để mảng không rỗng, nhưng mọi ngày trong khoảng bị đánh dấu
  `unreliable`; mẫu có cửa sổ đầu vào chạm các ngày này bị loại (xem splitting.py).
- Không bao giờ dùng quan trắc sau ngày đang xét (không nội suy hai chiều).
- Ngày đích thiếu không được điền làm nhãn: nhãn luôn lấy từ chuỗi quan trắc gốc (NaN → mặt nạ = 0).
"""
import numpy as np
import pandas as pd

from src.data.quality_control import runs_of, HN_COLS


def run_length(mask):
    r = np.zeros(len(mask), int)
    for i, L in runs_of(mask):
        r[i:i + L] = L
    return r


def age(mask):
    """Số ngày kể từ quan trắc hợp lệ gần nhất (0 nếu ngày đó có quan trắc)."""
    mask = np.asarray(mask, bool)
    a = np.zeros(len(mask))
    for i in range(1, len(mask)):
        a[i] = a[i - 1] + 1 if mask[i] else 0
    return a


def causal_impute(daily, max_gap):
    cdm, hnm = daily['cd_missing'].values, daily['hn_missing'].values
    assert not cdm[0] and not hnm[0], 'ngày đầu tiên phải có quan trắc để điền nhân quả'
    filled = daily.copy()
    cols = ['q_cd'] + HN_COLS
    filled[cols] = daily[cols].ffill()
    filled['cd_miss'], filled['hn_miss'] = cdm.astype(float), hnm.astype(float)
    filled['cd_age'], filled['hn_age'] = age(cdm), age(hnm)
    filled['cd_gap_len'], filled['hn_gap_len'] = run_length(cdm), run_length(hnm)
    filled['unreliable'] = (cdm & (filled['cd_gap_len'].values > max_gap)) | (hnm & (filled['hn_gap_len'].values > max_gap))
    assert not filled[cols].isna().any().any()
    return filled
