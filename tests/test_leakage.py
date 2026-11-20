"""Unit test chống rò rỉ dữ liệu và tính đúng đắn của bộ dữ liệu chuẩn (A3).

Chạy:  ~/miniconda3/envs/MTL/bin/python -m pytest -q tests/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_config, dataset_meta                                   # noqa: E402
from src.data.windowing import UnifiedDataset, load_daily                          # noqa: E402
from src.data.feature_engineering import feature_list                              # noqa: E402
from src.data.splitting import fold_names, split_info                              # noqa: E402
from src.data.quality_control import HN_COLS                                       # noqa: E402
from src.training.losses import flow_weights                                       # noqa: E402

CFG = load_config()
def _gap():
    """G của bộ dữ liệu đóng băng (manifest) nếu có, ngược lại theo kết quả chọn G (dataset_meta)."""
    import json
    from src.data.freeze import frozen_dir
    f = frozen_dir(CFG) / 'manifest.json'
    return json.loads(f.read_text(encoding='utf-8'))['max_gap'] if f.exists() else dataset_meta(CFG)['max_gap']


G = _gap()
SPLITS = fold_names(CFG) + [CFG['splits']['final']['name']]
FEATS = feature_list(CFG, 'F4') + ['qcd_ema3', 'hn_dmn', 'hn_bal', 'hn_ratio', 'hn_spill', 'qhn_rep_flag', 'qhn_rep_len']
FEATS = list(dict.fromkeys(FEATS))


@pytest.fixture(scope='module')
def daily():
    return load_daily(CFG, frozen=False)


@pytest.fixture(scope='module', params=[1, 10])
def ds(request, daily):
    return UnifiedDataset(CFG, request.param, G, daily)


def _perturb_after(daily, t_date, seed=0):
    """Nhân mọi giá trị số SAU ngày t với hệ số ngẫu nhiên (giữ nguyên vị trí NaN / cờ thiếu)."""
    rng = np.random.default_rng(seed)
    d = daily.copy()
    cols = ['q_cd'] + HN_COLS + ['mua_v1', 'mua_v2']
    m = d.index > t_date
    d.loc[m, cols] = d.loc[m, cols] * rng.uniform(0.2, 5.0, size=(m.sum(), len(cols)))
    return d


# ------------------------------------------------------------------ điền thiếu
def test_causal_fill_uses_only_past(daily):
    ds = UnifiedDataset(CFG, 1, G, daily)
    f, obs = ds.filled, daily
    for c in ['q_cd', 'hn_qxa', 'hn_qden', 'hn_mn']:
        assert np.allclose(obs[c].ffill(), f[c]), c
    # tuổi số liệu: số ngày kể từ quan trắc gần nhất ≤ t
    for col, age in [('cd_missing', 'cd_age'), ('hn_missing', 'hn_age')]:
        m, a = obs[col].values, f[age].values
        last = -1
        for t in range(len(m)):
            if not m[t]:
                last = t
            assert a[t] == t - last


def test_long_gaps_marked_unreliable(daily):
    ds = UnifiedDataset(CFG, 1, G, daily)
    f = ds.filled
    long_cd = f.cd_gap_len.values > G
    long_hn = f.hn_gap_len.values > G
    assert (f.unreliable.values == ((daily.cd_missing.values & long_cd) | (daily.hn_missing.values & long_hn))).all()
    # khoảng thiếu ≤ G không bị đánh dấu
    short = (daily.cd_missing.values & ~long_cd) | (daily.hn_missing.values & ~long_hn)
    assert not (f.unreliable.values & short & ~(daily.cd_missing.values & long_cd) & ~(daily.hn_missing.values & long_hn)).any()


def test_no_used_window_touches_long_gap(ds):
    unrel = ds.filled.unreliable.values
    W = CFG['imputation']['max_window']
    for sp in SPLITS:
        roles = ds.S[sp].values
        for i in np.where(roles != '')[0]:
            t = ds.S.pos.iloc[i]
            assert not unrel[t - W + 1:t + 1].any(), (sp, ds.S.date.iloc[i])


# ------------------------------------------------------------------ đặc trưng và cửa sổ
def test_rolling_rain_past_only(daily):
    ds = UnifiedDataset(CFG, 1, G, daily)
    r = daily.mua_v1.values
    for t in [30, 500, 1500, 2700]:
        assert np.isclose(ds.F.mua_v1_7.iloc[t], r[t - 6:t + 1].sum())
        assert np.isclose(ds.F.mua_v2_15.iloc[t], daily.mua_v2.values[t - 14:t + 1].sum())


@pytest.mark.parametrize('t_str,split', [('2021-06-15', 'F1'), ('2024-08-20', 'F3'), ('2025-07-20', 'FINAL'),
                                          ('2025-08-25', 'FINAL')])
def test_inputs_unchanged_when_future_perturbed(daily, t_str, split):
    """Phép thử nhiễu loạn: thay đổi MỌI dữ liệu sau ngày t → đầu vào của mẫu gốc t phải giữ nguyên."""
    t = pd.Timestamp(t_str)
    for H in (1, 10):
        a = UnifiedDataset(CFG, H, G, daily)
        b = UnifiedDataset(CFG, H, G, _perturb_after(daily, t))
        i = int(np.where(a.S.date.values == np.datetime64(t))[0][0])
        for L in (3, 30):
            va = a.view(split, FEATS, L, fut=False)
            vb = b.view(split, FEATS, L, fut=False)
            assert np.array_equal(va.X_seq[i], vb.X_seq[i]), (H, L, split)
            assert va.qlast[i] == vb.qlast[i]
        # và đích thì phải thay đổi (kiểm tra phép thử có tác dụng)
        assert not np.allclose(np.nan_to_num(a.QO[i]), np.nan_to_num(b.QO[i]))


def test_window_last_step_is_origin(ds):
    v = ds.view('FINAL', FEATS, 14)
    end = split_info(CFG, 'FINAL')[0]
    td = ds.train_days('FINAL')
    mu, sd = ds.T.loc[td, FEATS].mean(), ds.T.loc[td, FEATS].std()
    bi = ['hn_spill', 'cd_miss', 'hn_miss', 'qhn_rep_flag']
    mu[bi], sd[bi] = 0.0, 1.0
    for i in [0, 700, len(ds.S) - 1]:
        t = ds.S.pos.iloc[i]
        assert np.allclose(v.X_seq[i, -1], ((ds.T[FEATS].iloc[t] - mu) / sd).values, atol=1e-4)
        assert np.allclose(v.X_seq[i, 0], ((ds.T[FEATS].iloc[t - 13] - mu) / sd).values, atol=1e-4)


def test_targets_are_raw_observations(ds, daily):
    q = daily.q_cd.values
    for i in [0, 100, 1000, len(ds.S) - 1]:
        t = ds.S.pos.iloc[i]
        assert np.allclose(np.nan_to_num(ds.QO[i]), np.nan_to_num(q[t + 1:t + ds.H + 1]))
        assert (ds.M[i] == ~np.isnan(q[t + 1:t + ds.H + 1])).all()


# ------------------------------------------------------------------ chia tập
def test_all_labels_inside_partition(ds):
    """Mọi nhãn t+1 … t+H của một mẫu nằm trọn trong partition của mẫu đó (quan trọng cho H = 10)."""
    H = ds.H
    tdates = ds.S.date.values[:, None] + np.arange(1, H + 1) * np.timedelta64(1, 'D')
    for fd in CFG['splits']['folds']:
        r = ds.S[fd['name']].values
        assert (tdates[r == 'train'] <= np.datetime64(fd['train_end'])).all()
        v = tdates[r == 'val']
        assert (v >= np.datetime64(fd['val_start'])).all() and (v <= np.datetime64(fd['val_end'])).all()
        assert ds.S.date.values[r == 'val'].min() > np.datetime64(fd['train_end'])
    fn = CFG['splits']['final']
    r = ds.S['FINAL'].values
    assert (tdates[r == 'train'] <= np.datetime64(fn['train_end'])).all()
    assert (tdates[r == 'test'] > np.datetime64(fn['test_start'])).all()
    assert (tdates[r == 'test'] <= np.datetime64(CFG['period']['end'])).all()


def test_train_eval_targets_disjoint(ds):
    H = ds.H
    tg = lambda r, sp: set((ds.S.date.values[ds.S[sp].values == r][:, None] + np.arange(1, H + 1) * np.timedelta64(1, 'D')).ravel())
    for sp in SPLITS:
        ev = 'test' if sp == 'FINAL' else 'val'
        assert not (tg('train', sp) & tg(ev, sp)), sp
    assert not (tg('val', 'F3') & tg('test', 'FINAL'))


# ------------------------------------------------------------------ chuẩn hoá và ngưỡng
@pytest.mark.parametrize('split', SPLITS)
def test_scaler_and_quantiles_from_train_only(daily, split):
    end = split_info(CFG, split)[0]
    a = UnifiedDataset(CFG, 1, G, daily)
    b = UnifiedDataset(CFG, 1, G, _perturb_after(daily, end))
    va, vb = a.view(split, FEATS, 7), b.view(split, FEATS, 7)
    tr = va.idx_train
    assert np.array_equal(va.X_seq[tr], vb.X_seq[tr])                        # chuẩn hoá không đổi khi dữ liệu sau train đổi
    assert va.q95 == vb.q95 and va.q90 == vb.q90
    q = daily.q_cd[(daily.index <= end) & ~daily.cd_missing]
    assert np.isclose(va.q95, q.quantile(.95)) and np.isclose(va.q90, q.quantile(.90))
    wa = flow_weights(va.W_q, va.M, tr, va.q95, gamma=2)
    wb = flow_weights(vb.W_q, vb.M, tr, vb.q95, gamma=2)
    assert np.allclose(wa[tr], wb[tr])                                      # trọng số train chỉ phụ thuộc train


def test_same_samples_for_all_models(ds):
    v3, v30 = ds.view('F3', feature_list(CFG, 'F0'), 3), ds.view('F3', feature_list(CFG, 'F3'), 30)
    assert np.array_equal(v3.idx_train, v30.idx_train) and np.array_equal(v3.idx_eval, v30.idx_eval)
    assert np.allclose(v3.X_flat, v3.X_seq.reshape(len(v3.X_seq), -1))


def test_frozen_dataset_matches_manifest():
    from src.data.freeze import verify, frozen_dir
    if not (frozen_dir(CFG) / 'manifest.json').exists():
        pytest.skip('chưa đóng băng')
    man = verify(CFG)
    assert man['max_gap'] == G
