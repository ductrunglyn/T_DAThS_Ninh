"""Kiểm tra nhất quán dữ liệu: file gốc ↔ bảng ngày đã xử lý ↔ bộ dữ liệu đóng băng; đơn vị, miền giá trị; tập biến E1–E4;
quy tắc điền thiếu G ngày; số mẫu của từng tập. Không sửa dữ liệu — chỉ kiểm tra.

Chạy:  ~/miniconda3/envs/MTL/bin/python -m pytest -q tests/
Báo cáo kèm theo: docs/KiemTra_NhatQuan_DuLieu.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import load_config, path                                            # noqa: E402
from src.data.quality_control import build_daily, load_raw, RENAME                 # noqa: E402
from src.data.windowing import UnifiedDataset, load_daily                          # noqa: E402
from src.data.feature_engineering import feature_list                              # noqa: E402
from src.data.freeze import frozen_dir                                             # noqa: E402

CFG = load_config()
MAN = json.loads((frozen_dir(CFG) / 'manifest.json').read_text(encoding='utf-8'))
G = MAN['max_gap']
NUM = ['q_cd', 'hn_mn', 'hn_qden', 'hn_qnm', 'hn_qtran', 'hn_qxa', 'hn_gio', 'mua_v1', 'mua_v2',
       'cd_mn', 'cd_qnm', 'cd_qtran', 'cd_qxa', 'cd_gio']


@pytest.fixture(scope='module')
def processed():
    return pd.read_csv(path(CFG, 'processed', 'cuadat_huana_mua_daily.csv'), encoding='utf-8-sig',
                       parse_dates=['date'], index_col='date')


@pytest.fixture(scope='module')
def raw():
    return load_raw(CFG)


# ------------------------------------------------------------------ file gốc
@pytest.mark.parametrize('key', ['cuadat', 'huana'])
def test_raw_csv_equals_xlsx(key):
    """File .csv trong data/raw là bản chuyển đổi đúng của file .xlsx gốc (cùng ngày, cùng giá trị)."""
    pytest.importorskip('openpyxl')
    csv = path(CFG, 'raw', CFG['files'][key])
    x = pd.read_excel(csv.with_suffix('.xlsx'), header=4).rename(columns=RENAME)
    c = pd.read_csv(csv, encoding='utf-8-sig').rename(columns=RENAME)
    x['date'], c['date'] = pd.to_datetime(x['date']), pd.to_datetime(c['date'])
    assert x['date'].equals(c['date'])
    for col in ['mn', 'qden', 'qnm', 'qtran', 'qxa', 'gio']:
        assert np.allclose(x[col].astype(float), c[col].astype(float), equal_nan=True), col


# ------------------------------------------------------------------ bảng ngày đã xử lý
def test_processed_equals_rebuild_and_frozen(processed):
    """Bảng ngày của notebook 01 = dựng lại từ file gốc = bản đóng băng v1 (daily_clean.csv)."""
    rebuilt, _ = build_daily(CFG)
    frozen = load_daily(CFG, frozen=True)
    for other in (rebuilt, frozen):
        assert other.index.equals(processed.index)
        for c in NUM:
            assert np.allclose(other[c].astype(float), processed[c].astype(float), equal_nan=True), c
        for c in ['cd_missing', 'hn_missing']:
            assert (other[c].astype(bool).values == processed[c].astype(bool).values).all(), c


def test_period_and_missing_days(processed):
    assert processed.index[0] == CFG['period']['start'] and processed.index[-1] == CFG['period']['end']
    assert len(processed) == 2781 and processed.index.to_series().diff().dropna().eq(pd.Timedelta(days=1)).all()
    assert int(processed.cd_missing.sum()) == 139 and int(processed.hn_missing.sum()) == 140


def test_values_trace_to_raw(processed, raw):
    """Mọi giá trị dùng làm đầu vào/đích lấy nguyên từ file gốc (trừ Q ra Hủa Na tính lại và 2 mực nước CĐ ngoại lai)."""
    cd, hn, rain = raw
    d = processed
    ok = ~d.cd_missing
    assert np.allclose(d.q_cd[ok], cd.loc[d.index[ok], 'qden'])
    ok = ~d.hn_missing
    for c in ['mn', 'qden', 'qnm', 'qtran', 'gio']:
        assert np.allclose(d[f'hn_{c}'][ok], hn.loc[d.index[ok], c]), c
    assert np.allclose(d.mua_v1, rain.loc[d.index, 'Vung_1_Hua_Na'])
    assert np.allclose(d.mua_v2, rain.loc[d.index, 'Vung_2_Giua_HuaNa_CuaDat'])


def test_q_out_huana_is_turbine_plus_spill(processed, raw):
    """Q ra Hủa Na = Q nhà máy + Q tràn (tính lại); cột tổng xả gốc lệch > 0,5 m³/s ở đúng 74 ngày trong giai đoạn nghiên cứu."""
    d = processed[~processed.hn_missing]
    assert np.allclose(d.hn_qxa, d.hn_qnm + d.hn_qtran)
    hn = raw[1].loc[CFG['period']['start']:CFG['period']['end']]
    assert int(((hn['qxa'] - (hn['qnm'] + hn['qtran'])).abs() > 0.5).sum()) == 74


def test_units_and_ranges(processed):
    """Lưu lượng m³/s ≥ 0; mưa mm/ngày ≥ 0; giờ phát điện trong [0, 24]; mực nước CĐ ≤ 120 m sau làm sạch."""
    d = processed
    for c in ['q_cd', 'hn_qden', 'hn_qnm', 'hn_qtran', 'hn_qxa', 'cd_qnm', 'cd_qtran', 'cd_qxa', 'mua_v1', 'mua_v2']:
        assert (d[c].dropna() >= 0).all(), c
    for c in ['hn_gio', 'cd_gio']:
        assert d[c].dropna().between(0, 24).all(), c
    assert d.cd_mn.max() <= CFG['quality']['cd_level_outlier']
    assert d.q_cd.max() == 4234.0 and d.q_cd.idxmax() == pd.Timestamp('2025-08-26')


def test_known_huana_level_spikes_documented(processed):
    """Mực nước Hủa Na lệch > 10 m so với trung vị ±3 ngày chỉ ở 2 ngày đã ghi trong docs/KiemTra_NhatQuan_DuLieu.md
    (giữ nguyên trong v1). Nếu dữ liệu thay đổi, test này báo để cập nhật báo cáo."""
    z = processed.hn_mn
    dev = (z - z.rolling(7, center=True, min_periods=3).median()).abs()
    assert sorted(str(t.date()) for t in dev[dev > 10].index) == ['2019-05-08', '2020-07-06']


# ------------------------------------------------------------------ tập biến E1–E4
def test_feature_sets_E1_to_E4():
    base = feature_list(CFG, 'F0')
    assert not any(c.startswith(('hn_', 'qhn_')) for c in base) and len(base) == 13
    flags, rep = ['hn_miss', 'hn_age'], ['qhn_rep_flag', 'qhn_rep_len']
    assert feature_list(CFG, 'F1') == base + ['hn_qxa'] + flags                         # E2: Q ra
    assert feature_list(CFG, 'F2') == base + ['hn_qden'] + flags + rep                  # E3: Q đến
    assert feature_list(CFG, 'F3') == base + ['hn_qxa', 'hn_qden', 'hn_mn'] + flags + rep   # E4: Q ra + Q đến + Z
    fs = json.loads((frozen_dir(CFG) / 'feature_sets.json').read_text(encoding='utf-8'))
    assert all(fs[s] == feature_list(CFG, s) for s in CFG['feature_sets'])


# ------------------------------------------------------------------ điền thiếu, biến đổi, mẫu
@pytest.fixture(scope='module', params=[1, 10])
def ds(request):
    return UnifiedDataset(CFG, request.param, G, load_daily(CFG))


def test_used_windows_hold_at_most_G_days(ds):
    """Trong cửa sổ 30 ngày của mọi mẫu được dùng, giá trị điền nhân quả không 'giữ' quá G ngày."""
    W = CFG['imputation']['max_window']
    age_cd, age_hn = ds.filled.cd_age.values, ds.filled.hn_age.values
    used = (ds.S[[f['name'] for f in CFG['splits']['folds']] + ['FINAL']] != '').any(axis=1).values
    for t in ds.S.pos.values[used]:
        assert age_cd[t - W + 1:t + 1].max() <= G and age_hn[t - W + 1:t + 1].max() <= G


def test_log_transform_of_flows(ds):
    f, T = ds.F, ds.T
    for c in ['q_cd', 'hn_qxa', 'hn_qden', 'mua_v1_7']:
        assert np.allclose(T[c], np.log1p(f[c]), equal_nan=True), c
    assert np.allclose(T['hn_mn'], f['hn_mn'], equal_nan=True)                     # mực nước không lấy log


def test_sample_counts_match_split_report(ds):
    rep = pd.read_csv(frozen_dir(CFG) / 'split_report.csv', encoding='utf-8-sig', index_col=[0, 1])
    for sp in [f['name'] for f in CFG['splits']['folds']] + ['FINAL']:
        ev = 'test' if sp == 'FINAL' else 'val'
        r = rep.loc[(f'H{ds.H}', sp)]
        assert (ds.S[sp] == 'train').sum() == r['train: số mẫu'] and (ds.S[sp] == ev).sum() == r['đánh giá: số mẫu'], sp
