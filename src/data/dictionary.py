"""Từ điển dữ liệu (A1): biến gốc và đặc trưng — đơn vị, nguồn, cách tính, thời điểm sẵn sàng, mức độ đã xác minh.

Nguyên tắc: chỉ ghi điều có căn cứ trong file nguồn hoặc kiểm tra được trên dữ liệu; điều không có tài liệu thì ghi
"CHƯA XÁC MINH" và nêu giả định đang dùng.
"""
import pandas as pd

from src.data.feature_engineering import FEATURE_META

NV = 'CHƯA XÁC MINH'
DAY = ('1 ngày theo "Báo cáo trung bình ngày" (tiêu đề file gốc); giờ bắt đầu/kết thúc của ngày: ' + NV)
AVAIL_OP = ('Giả định: có sau khi kết thúc ngày t, trước thời điểm phát hành dự báo t+1. ' + NV +
            ' (cần xác nhận thời điểm đơn vị vận hành chốt và gửi báo cáo ngày)')

RAW = [
    # biến, cột gốc, file, đơn vị, cách tính / ý nghĩa, đại diện, sẵn sàng, đã xác minh, dùng làm đầu vào
    ('q_cd', 'Lưu lượng đến hồ (m³/s)', 'vanhanh_cuadat', 'm³/s',
     'Lưu lượng đến hồ Cửa Đạt. File không ghi phương pháp. Giả thuyết: tính ngược từ cân bằng nước (ΔV/Δt + Q xả); '
     'dao động "răng cưa" (tự tương quan ΔQ = −0,22) phù hợp giả thuyết này. Phương pháp: ' + NV,
     DAY, AVAIL_OP, 'Đơn vị, giá trị khớp file gốc: đã kiểm tra. Phương pháp tính: ' + NV, 'Đích + lịch sử (≤ t)'),
    ('hn_qden', 'Lưu lượng đến hồ (m³/s)', 'Vanhanh_huaNa', 'm³/s',
     'Lưu lượng đến hồ Hủa Na. Có 40 đoạn giá trị tròn không đổi ≥ 5 ngày (294 ngày, chủ yếu T1–T4) — có thể là giá trị '
     'ước lượng. Phương pháp tính: ' + NV, DAY, AVAIL_OP, NV, 'E3, E4 (≤ t)'),
    ('hn_qxa', 'Lưu lượng qua nhà máy + Lưu lượng xả qua cửa tràn', 'Vanhanh_huaNa', 'm³/s',
     'Q ra Hủa Na = Q nhà máy + Q tràn (tính lại; cột "Tổng lưu lượng xả" gốc lệch ở 74 ngày, có ngày thiếu hẳn phần '
     'nhà máy). Kế hoạch / dự báo xả t+1: KHÔNG có trong dữ liệu.', DAY, AVAIL_OP,
     'Công thức tính lại: đã kiểm tra trên dữ liệu. Thời điểm sẵn sàng: ' + NV, 'E2, E4 (≤ t)'),
    ('hn_qnm', 'Lưu lượng qua nhà máy (m³/s)', 'Vanhanh_huaNa', 'm³/s', 'Q qua tổ máy Hủa Na', DAY, AVAIL_OP, NV, 'F4 (phụ)'),
    ('hn_qtran', 'Lưu lượng xả qua cửa tràn (m³/s)', 'Vanhanh_huaNa', 'm³/s', 'Q qua tràn Hủa Na', DAY, AVAIL_OP, NV, 'F4 (phụ)'),
    ('hn_mn', 'Mực nước hồ (m)', 'Vanhanh_huaNa', 'm', 'Mực nước hồ Hủa Na (trung bình ngày hay thời điểm: ' + NV + ')',
     DAY, AVAIL_OP, NV, 'E4 (≤ t)'),
    ('hn_gio', 'Số giờ phát điện', 'Vanhanh_huaNa', 'giờ',
     'Không nhất quán với Q nhà máy: 62 ngày (01/2019–08/2026) giờ = 0 nhưng Q nhà máy > 0 (Cửa Đạt: 294 ngày). Độ tin cậy thấp.',
     DAY, AVAIL_OP, 'Mâu thuẫn đã phát hiện; nguyên nhân: ' + NV, 'F4 (phụ)'),
    ('mua_v1', 'Vung_1_Hua_Na', 'CHIRPS_FINAL_2VUNG', 'mm/ngày',
     'Mưa CHIRPS trung bình vùng trên lưu vực Hủa Na. Ranh giới vùng, cách lấy trung bình, phiên bản CHIRPS: ' + NV,
     'Ngày theo quy ước của sản phẩm CHIRPS; độ lệch so với ngày báo cáo vận hành: ' + NV,
     'Giả định: có tại t. Thực tế sản phẩm CHIRPS được phát hành sau một độ trễ; khi vận hành thực cần thay bằng mưa đo '
     'hoặc mưa ước lượng thời gian thực. ' + NV, NV, 'mọi kịch bản (≤ t); FUT: t+1 (cận trên)'),
    ('mua_v2', 'Vung_2_Giua_HuaNa_CuaDat', 'CHIRPS_FINAL_2VUNG', 'mm/ngày',
     'Mưa CHIRPS trung bình vùng trên khu giữa Hủa Na – Cửa Đạt. ' + NV, 'như mua_v1', 'như mua_v1', NV,
     'mọi kịch bản (≤ t); FUT: t+1 (cận trên)'),
]


def raw_dictionary():
    return pd.DataFrame(RAW, columns=['biến', 'cột trong file gốc', 'file', 'đơn vị', 'cách tính / ý nghĩa',
                                      'khoảng thời gian đại diện', 'thời điểm sẵn sàng', 'đã xác minh?', 'dùng làm']).set_index('biến')


RULE = {'log1p': 'log(1 + x), rồi z-score theo train của fold', 'slog': 'sign(x)·log(1 + |x|), rồi z-score',
        'none': 'z-score theo train của fold', 'binary': 'giữ 0/1'}
CREATE = {
    'qcd_ema3': 'EMA span 3 của Q CĐ đã điền nhân quả (chỉ dùng ≤ t)',
    'mua_v1_3': 'Σ mưa V1 t−2 … t', 'mua_v1_7': 'Σ mưa V1 t−6 … t', 'mua_v1_15': 'Σ mưa V1 t−14 … t',
    'mua_v2_3': 'Σ mưa V2 t−2 … t', 'mua_v2_7': 'Σ mưa V2 t−6 … t', 'mua_v2_15': 'Σ mưa V2 t−14 … t',
    'doy_sin': 'sin(2π·ngày trong năm/365,25)', 'doy_cos': 'cos(2π·ngày trong năm/365,25)',
    'hn_dmn': 'Z(t) − Z(t−1)', 'hn_bal': 'Q đến(t) − Q ra(t)', 'hn_ratio': 'Q ra(t)/(Q đến(t) + 1), cắt ở 10',
    'hn_spill': '1 nếu Q tràn(t) > 50 m³/s', 'cd_miss': '1 nếu Q CĐ ngày t thiếu (giá trị đang được giữ từ ngày trước)',
    'cd_age': 'số ngày kể từ quan trắc Q CĐ gần nhất ≤ t', 'hn_miss': '1 nếu số liệu Hủa Na ngày t thiếu',
    'hn_age': 'số ngày kể từ quan trắc Hủa Na gần nhất ≤ t',
    'qhn_rep_len': 'số ngày liên tiếp đến t mà Q đến HN = giá trị ngày trước (ngày thiếu ngắt chuỗi)',
    'qhn_rep_flag': '1 nếu qhn_rep_len ≥ 4 (≥ 5 giá trị bằng nhau)'}


def feature_dictionary(cfg, max_gap):
    rows = []
    for k, (u, t, g, d) in FEATURE_META.items():
        rows.append({'đặc trưng': k, 'nhóm': g, 'đơn vị': u, 'mô tả': d,
                     'quy tắc tạo': CREATE.get(k, 'giá trị ngày t của biến gốc (đã điền nhân quả)'),
                     'biến đổi': RULE[t], 'dữ liệu dùng': '≤ t (đặc trưng tại ngày t chỉ dùng ngày ≤ t)',
                     'xử lý thiếu': f'forward fill khoảng thiếu ≤ {max_gap} ngày; mẫu chạm khoảng thiếu dài hơn bị loại',
                     'tập biến': ', '.join(s for s in cfg['feature_sets'] if k in _resolve(cfg, s))})
    return pd.DataFrame(rows).set_index('đặc trưng')


def _resolve(cfg, s):
    out = []
    for it in cfg['feature_sets'][s]:
        out += cfg['feature_groups'].get(it, [it])
    return out
