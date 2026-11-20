"""Đọc dữ liệu gốc, làm sạch, hợp nhất thành bảng ngày (GIỮ NGUYÊN ngày thiếu = NaN) và các kiểm tra chất lượng.

Không nội suy, không chuẩn hoá, không tạo đặc trưng cho mô hình ở đây.
"""
import numpy as np
import pandas as pd

from src.utils import path

RENAME = {'Ngày': 'date', 'Mực nước hồ (m)': 'mn', 'Lưu lượng đến hồ (m³/s)': 'qden',
          'Lưu lượng qua nhà máy (m³/s)': 'qnm', 'Lưu lượng xả qua cửa tràn (m³/s)': 'qtran',
          'Tổng lưu lượng xả (m³/s)': 'qxa', 'Số giờ phát điện': 'gio', 'Ghi chú': 'ghichu'}
HN_COLS = ['hn_mn', 'hn_qden', 'hn_qnm', 'hn_qtran', 'hn_qxa', 'hn_gio']
CD_REF = ['cd_mn', 'cd_qnm', 'cd_qtran', 'cd_qxa', 'cd_gio']


def load_operation(p):
    d = pd.read_csv(p, encoding='utf-8-sig').rename(columns=RENAME)
    d['date'] = pd.to_datetime(d['date'], format='%Y-%m-%d')
    assert not d['date'].duplicated().any(), f'ngày trùng trong {p}'
    return d.set_index('date').sort_index().drop(columns='ghichu')


def load_rain(p):
    r = pd.read_csv(p, encoding='utf-8-sig')
    r['date'] = pd.to_datetime(r['date'], format='%m/%d/%Y')
    return r.set_index('date').sort_index()


def load_raw(cfg):
    raw = path(cfg, 'raw')
    f = cfg['files']
    return (load_operation(raw / f['cuadat']), load_operation(raw / f['huana']), load_rain(raw / f['rain']))


def build_daily(cfg):
    """Bảng ngày hợp nhất. Trả về (daily, log) — log là danh sách các thao tác làm sạch đã thực hiện."""
    cd_raw, hn_raw, rain_raw = load_raw(cfg)
    a, b = cfg['period']['start'], cfg['period']['end']
    cd, hn = cd_raw.loc[a:b].copy(), hn_raw.loc[a:b].copy()
    rain = rain_raw.loc[a:b, ['Vung_1_Hua_Na', 'Vung_2_Giua_HuaNa_CuaDat']].set_axis(['mua_v1', 'mua_v2'], axis=1)
    log = []
    bad = (hn['qxa'] - (hn['qnm'] + hn['qtran'])).abs() > 0.5
    log.append(f'Hủa Na: tổng xả ≠ nhà máy + tràn ở {int(bad.sum())} ngày → tính lại Q ra = Q nhà máy + Q tràn')
    hn['qxa'] = hn['qnm'] + hn['qtran']
    out = cd['mn'] > cfg['quality']['cd_level_outlier']
    log.append(f'Cửa Đạt: {int(out.sum())} ngày mực nước > {cfg["quality"]["cd_level_outlier"]} m '
               f'({", ".join(f"{d.date()}: {v}" for d, v in cd.loc[out, "mn"].items())}) → đặt thiếu')
    cd.loc[out, 'mn'] = np.nan
    for nm, d in [('Cửa Đạt', cd), ('Hủa Na', hn)]:
        neg = int((d[['qden', 'qnm', 'qtran', 'qxa']] < 0).sum().sum())
        hrs = int(((d['gio'] < 0) | (d['gio'] > 24)).sum())
        log.append(f'{nm}: giá trị âm {neg}, giờ phát điện ngoài [0, 24] {hrs}, ô trống {int(d.isna().sum().sum())}')

    idx = pd.date_range(a, b, freq='D', name='date')
    assert rain.index.equals(idx), 'mưa CHIRPS phải liên tục'
    cd, hn = cd.reindex(idx), hn.reindex(idx)
    daily = pd.DataFrame(index=idx)
    daily['q_cd'] = cd['qden']
    for c in ['mn', 'qden', 'qnm', 'qtran', 'qxa', 'gio']:
        daily[f'hn_{c}'] = hn[c]
    daily['mua_v1'], daily['mua_v2'] = rain['mua_v1'], rain['mua_v2']
    for c in ['mn', 'qnm', 'qtran', 'qxa', 'gio']:
        daily[f'cd_{c}'] = cd[c]
    daily['cd_missing'] = daily['q_cd'].isna()
    daily['hn_missing'] = daily['hn_qden'].isna()
    # ngày thiếu Hủa Na: thiếu cả dòng (mọi cột cùng thiếu)
    assert (daily[HN_COLS].isna().any(axis=1) == daily['hn_missing']).all()
    return daily, log


# ---------------------------------------------------------------- khoảng thiếu
def runs_of(mask):
    """Danh sách (vị trí bắt đầu, độ dài) của các đoạn True liên tiếp."""
    mask = np.asarray(mask, bool)
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j - i)); i = j
        else:
            i += 1
    return out


def gap_table(daily):
    rows = []
    for nm, col in [('Cửa Đạt', 'cd_missing'), ('Hủa Na', 'hn_missing')]:
        for i, L in runs_of(daily[col].values):
            rows.append({'trạm': nm, 'từ': daily.index[i].date(), 'đến': daily.index[i + L - 1].date(), 'số ngày': L,
                         'loại': '1 ngày' if L == 1 else ('2–3 ngày' if L <= 3 else ('4–7 ngày' if L <= 7 else '> 7 ngày'))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- giá trị lặp (nhân quả)
def repeat_length_causal(s):
    """Số ngày liên tiếp (tính đến ngày t) mà giá trị bằng giá trị ngày trước — chỉ dùng dữ liệu ≤ t.
    0 nếu giá trị đổi hoặc thiếu."""
    v = np.asarray(s, float)
    out = np.zeros(len(v))
    for t in range(1, len(v)):
        if not np.isnan(v[t]) and not np.isnan(v[t - 1]) and v[t] == v[t - 1]:
            out[t] = out[t - 1] + 1
    return out


def repeat_runs(s, min_len):
    """Các đoạn giá trị không đổi ≥ min_len ngày (phân tích, không dùng làm đầu vào)."""
    v = pd.Series(s)
    same = v.diff() == 0
    grp = (~same).cumsum()
    L = same.groupby(grp).sum() + 1
    rows = []
    for g, n in L[L >= min_len].items():
        seg = v[grp == g]
        rows.append({'từ': seg.index[0].date(), 'số ngày': int(n), 'giá trị': seg.iloc[0], 'tháng': seg.index[0].month})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- răng cưa Q đến Cửa Đạt
def sawtooth_table(daily, jump=0.30):
    """Ngày Q đến CĐ tăng/giảm > jump rồi đảo chiều ngày sau, kèm các căn cứ đối chiếu.
    Dùng t+1 nên CHỈ để phân tích chất lượng, không làm đầu vào."""
    q = daily['q_cd']
    d1 = q.diff()
    saw = (d1 * d1.shift(-1) < 0) & (d1.abs() > jump * q) & (d1.shift(-1).abs() > jump * q)
    rows = []
    for t in np.where(saw.values)[0]:
        if t < 2 or t + 1 >= len(q):
            continue
        d = daily.index[t]
        rain2 = daily['mua_v2'].iloc[t - 2:t + 1].max()
        rain1 = daily['mua_v1'].iloc[t - 2:t + 1].max()
        hq = daily['hn_qxa']
        dhn = np.log1p(hq.iloc[t]) - np.log1p(hq.iloc[t - 1]) if not (np.isnan(hq.iloc[t]) or np.isnan(hq.iloc[t - 1])) else np.nan
        dz = daily['cd_mn'].iloc[t] - daily['cd_mn'].iloc[t - 1]
        dzn = daily['cd_mn'].iloc[t + 1] - daily['cd_mn'].iloc[t]
        dxa = daily['cd_qxa'].iloc[t] - daily['cd_qxa'].iloc[t - 1]
        rows.append({'ngày': d.date(), 'Q(t−1)': q.iloc[t - 1], 'Q(t)': q.iloc[t], 'Q(t+1)': q.iloc[t + 1],
                     'hướng': 'đỉnh' if d1.iloc[t] > 0 else 'đáy', 'mưa V1 max (t−2…t)': rain1, 'mưa V2 max (t−2…t)': rain2,
                     'Δlog Q ra HN (t)': dhn, 'ΔZ CĐ (t) cm': 100 * dz, 'ΔZ CĐ (t+1) cm': 100 * dzn, 'ΔQ xả CĐ (t)': dxa})
    tab = pd.DataFrame(rows)
    if len(tab):
        has_cause = (tab['mưa V2 max (t−2…t)'] >= 10) | (tab['mưa V1 max (t−2…t)'] >= 10) | (tab['Δlog Q ra HN (t)'].abs() >= 0.3)
        tab['đánh giá'] = np.where(has_cause, 'có căn cứ thủy văn (mưa ≥ 10 mm hoặc Q ra HN đổi ≥ 35 %)',
                                   'nghi nhiễu tính toán (không có mưa / thay đổi xả HN tương ứng)')
    return tab


# ---------------------------------------------------------------- kiểm tra tổng hợp
def lag_corr(x, y, ok, lags):
    out = {}
    for k in lags:
        xs = x.shift(k)
        m = ok & xs.notna() & y.notna()
        out[k] = np.corrcoef(xs[m], y[m])[0, 1]
    return out


def quality_audit(cfg, daily):
    aud = {}
    cd_src, _, _ = load_raw(cfg)
    rr = ~daily['cd_missing']
    cm = cd_src.index.intersection(daily.index[rr])
    aud['1. Q đến CĐ khớp file gốc'] = f'{np.allclose(cd_src.loc[cm, "qden"], daily.loc[cm, "q_cd"])} ({len(cm)} ngày)'
    dq = np.log1p(daily['q_cd']).diff()
    ok2 = rr & rr.shift(1, fill_value=False)
    for v in ['mua_v1', 'mua_v2']:
        cc = lag_corr(daily[v], dq, ok2, range(-2, 4))
        aud[f'2. corr({v}(t−k), Δlog Q CĐ(t)), k = −2…3'] = ', '.join(f'{k}: {c:+.3f}' for k, c in cc.items()) + \
            f' → cực đại tại k = {max(cc, key=cc.get)}'
    hm = ~daily['hn_missing']
    ok3 = ok2 & hm & hm.shift(1, fill_value=False)
    cc = lag_corr(np.log1p(daily['hn_qxa']).diff(), dq, ok3, range(-2, 4))
    aud['3. corr(Δlog Q ra HN(t−k), Δlog Q CĐ(t)), k = −2…3'] = ', '.join(f'{k}: {c:+.3f}' for k, c in cc.items())
    d1 = daily['q_cd'].diff()
    aud['4. tự tương quan bậc 1 của ΔQ CĐ'] = f'{d1[ok2].autocorr(1):+.3f} (âm → dao động đảo chiều)'
    st = sawtooth_table(daily, cfg['quality']['sawtooth_jump'])
    aud['4. số ngày "răng cưa"'] = f'{len(st)} / {int(rr.sum())} ngày'
    if len(st):
        aud['4. răng cưa nghi nhiễu tính toán'] = f'{int(st["đánh giá"].str.startswith("nghi").sum())} / {len(st)}'
    aud['4. Q đến CĐ < 5 m³/s'] = f'{int((daily.q_cd[rr] < 5).sum())} ngày, nhỏ nhất {daily.q_cd[rr].min():.1f}'
    for col, nm in [('hn_qden', 'Q đến HN'), ('hn_qxa', 'Q ra HN'), ('q_cd', 'Q đến CĐ'), ('hn_mn', 'Mực nước HN')]:
        r = repeat_runs(daily[col], cfg['quality']['repeat_min'])
        aud[f'5. đoạn {nm} không đổi ≥ {cfg["quality"]["repeat_min"]} ngày'] = f'{len(r)} đoạn, {int(r["số ngày"].sum()) if len(r) else 0} ngày'
    return pd.Series(aud, name='kết quả').to_frame()


# ---------------------------------------------------------------- bổ sung (A2)
def operation_hours_check(cfg):
    """Đối chiếu số giờ phát điện với lưu lượng qua nhà máy trong file gốc."""
    cd, hn, _ = load_raw(cfg)
    rows = []
    for nm, d in [('Cửa Đạt', cd), ('Hủa Na', hn)]:
        d = d.loc[cfg['period']['start']:cfg['period']['end']]
        rows.append({'hồ': nm, 'số ngày': len(d), 'giờ = 0': int((d.gio == 0).sum()),
                     'giờ = 0 nhưng Q nhà máy > 0': int(((d.gio == 0) & (d.qnm > 0)).sum()),
                     'giờ > 0 nhưng Q nhà máy = 0': int(((d.gio > 0) & (d.qnm == 0)).sum()),
                     'giờ = 0 và Q nhà máy = 0': int(((d.gio == 0) & (d.qnm == 0)).sum())})
    return pd.DataFrame(rows).set_index('hồ')


def nan_replacements(cfg, daily):
    """Mọi ô bị đặt thành NaN hoặc thiếu so với lịch ngày liên tục: (nguồn, ngày, cột, giá trị gốc, lý do)."""
    cd_raw, hn_raw, _ = load_raw(cfg)
    a, b = cfg['period']['start'], cfg['period']['end']
    rows = []
    out = cd_raw.loc[a:b, 'mn'] > cfg['quality']['cd_level_outlier']
    for d, v in cd_raw.loc[a:b].loc[out, 'mn'].items():
        rows.append({'nguồn': 'Cửa Đạt', 'ngày': d.date(), 'cột': 'mực nước', 'giá trị gốc': v, 'lý do': 'ngoại lai > 120 m'})
    for nm, raw, col in [('Cửa Đạt', cd_raw, 'cd_missing'), ('Hủa Na', hn_raw, 'hn_missing')]:
        absent = daily.index[daily[col].values]
        for d in absent:
            rows.append({'nguồn': nm, 'ngày': d.date(), 'cột': 'cả dòng', 'giá trị gốc': None,
                         'lý do': 'ngày không có trong file gốc' if d not in raw.index else 'ô trống trong file gốc'})
    return pd.DataFrame(rows)
