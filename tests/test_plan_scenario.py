"""Kịch bản bổ sung KH (kế hoạch xả Hủa Na ngày t+1): đúng ngày, không làm thay đổi E2, sai số kế hoạch đúng thiết kế,
dữ liệu đóng băng không bị đụng tới."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import load_config                                   # noqa: E402
from src.data.freeze import verify                                  # noqa: E402
from src.data.preprocessing import Scaler                           # noqa: E402
from src.data.feature_engineering import feature_list               # noqa: E402
from src import pipeline, plan_scenario as K                        # noqa: E402
from src import results_store as S                                  # noqa: E402

CFG = load_config()
SPLITS = ['F1', 'F2', 'F3', 'FINAL']


@pytest.fixture(scope='module')
def ds():
    return pipeline.get_dataset(CFG, 1)


def test_frozen_dataset_still_verifies():
    verify(CFG)                                     # src/data/ không bị sửa, manifest v1 vẫn khớp


@pytest.mark.parametrize('split', SPLITS)
def test_plan_is_hn_outflow_of_next_day(ds, split):
    v = ds.view(split, feature_list(CFG, 'F1'), 3)
    vp = K.with_plan(ds, v)
    z = Scaler(ds.T, ds.train_days(split))(ds.T)['hn_qxa'].values
    pos = ds.S['pos'].values
    assert vp.fut.shape == (len(pos), 1)
    assert np.allclose(vp.fut[:, 0], z[pos + 1], atol=1e-5)          # đúng ngày t+1
    j = v.feats.index('hn_qxa')
    assert np.allclose(v.X_seq[:, -1, j], z[pos], atol=1e-5)         # cửa sổ E2 vẫn dừng ở ngày t
    assert v.fut is None and vp.X_flat.shape[1] == v.X_flat.shape[1] + 1


def test_plan_does_not_alter_targets_or_samples(ds):
    v = ds.view('FINAL', feature_list(CFG, 'F1'), 3)
    vp = K.with_plan(ds, v, 0.3, 1)
    for a in ['X_seq', 'Y', 'M', 'QO', 'idx_train', 'idx_eval']:
        assert np.array_equal(getattr(v, a), getattr(vp, a), equal_nan=True), a


def test_noise_common_random_numbers_and_seed(ds):
    q = ds.filled['hn_qxa'].values
    ok = q > 0
    z = {}
    for s in (0.1, 0.3):
        qs = np.expm1(K.plan_series(ds, s, 0))
        keep = ok & (qs > 0)                         # bỏ ô bị chặn ở 0
        z[s] = np.where(keep, (qs / np.where(ok, q, 1) - 1) / s, np.nan)
    both = ~np.isnan(z[0.1]) & ~np.isnan(z[0.3])
    assert np.allclose(z[0.1][both], z[0.3][both], atol=1e-6)        # cùng chuỗi z cho mọi σ
    assert not np.allclose(K.plan_series(ds, 0.2, 0), K.plan_series(ds, 0.2, 1))   # khác hạt giống → khác chuỗi
    assert np.allclose(K.plan_series(ds, 0.0, 5), np.log1p(q))       # σ = 0: đúng lưu lượng thực đo
    assert (np.expm1(K.plan_series(ds, 0.5, 2)) >= 0).all()


def test_saved_predictions_share_target_days_with_E2():
    kh = S.read_pred(CFG, 'test', 1, 'KH')
    if not len(kh):
        pytest.skip('chưa chạy notebook 09')
    for m in K.MODELS:
        e2 = S.read_pred(CFG, 'test', 1, 'chinh', m, 'E2')
        e2 = e2[e2.seed == e2.seed.min()].set_index('ngay_goc')
        for e, d in kh[kh.model == m].groupby('exp'):
            d = d[d.seed == d.seed.min()].set_index('ngay_goc')
            assert d.index.equals(e2.index), (m, e)
            assert np.allclose(d.Q_do, e2.Q_do, equal_nan=True), (m, e)
        n_seed = kh[kh.model == m].groupby('exp').seed.nunique()
        assert n_seed.drop('KH_TD').eq(len(CFG['seeds']['final'])).all()
