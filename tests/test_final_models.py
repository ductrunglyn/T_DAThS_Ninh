"""Tái lập: weight tốt nhất đã lưu (results/<baselines|models>/<mô hình>/final_models: E2, t+1, huấn luyện 2019–2024)
nạp lại trên bộ dữ liệu đóng băng phải cho đúng dự báo test đã lưu trong results/du_bao.csv.gz và đúng chỉ tiêu trong
results/ket_qua_thuc_nghiem.csv.

Pipeline chạy mạng hồi tiếp trên GPU (cuDNN), test này chạy trên CPU: sai khác float32 đo được ≤ 0,054 % → ngưỡng 0,1 %.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import load_config                                   # noqa: E402
from src.data.windowing import UnifiedDataset, load_daily           # noqa: E402
from src.models import rnn, lstm, gru                               # noqa: E402
from src.pipeline import out_dir                                    # noqa: E402
from src.evaluation.metrics import summary                          # noqa: E402
from src import results_store as S                                  # noqa: E402

CFG = load_config()
DEEP = {'rnn': rnn, 'lstm': lstm, 'gru': gru}
MODELS = ['ridge', 'extra_trees', 'xgboost', 'rnn', 'lstm', 'gru']
_DS = {}


def _dataset(H, G):
    if H not in _DS:
        _DS[H] = UnifiedDataset(CFG, H, G, load_daily(CFG))
    return _DS[H]


def _predict(meta, f, v):
    idx = v.idx_eval
    if meta['model'] in DEEP:
        net = DEEP[meta['model']].build(len(meta['feats']), meta['hp']['hidden'], meta['hp']['layers'],
                                        CFG['deep']['dropout'], meta['H'])
        net.load_state_dict(torch.load(f, map_location='cpu'))
        net.eval()
        with torch.no_grad():
            return v.to_q(net(torch.as_tensor(v.X_seq[idx], dtype=torch.float32)).numpy())
    return joblib.load(f).predict(v, idx)


def _meta(model):
    f = out_dir(CFG, model) / 'final_models' / 'mo_hinh.json'
    if not f.exists():
        pytest.skip('chưa có weight đã lưu')
    return json.loads(f.read_text(encoding='utf-8'))


@pytest.mark.parametrize('model', MODELS)
def test_only_best_weights_are_kept(model):
    """Chỉ lưu weight tốt nhất: cấu hình chính E2, t+1; đủ hạt giống của huấn luyện cuối; không có file thừa."""
    allm = _meta(model)
    assert list(allm) == ['E2'] and allm['E2']['H'] == 1
    seeds = [0] if model == 'ridge' else CFG['seeds']['final']
    assert sorted(map(int, allm['E2']['files'])) == seeds
    d = out_dir(CFG, model) / 'final_models'
    assert sorted(p.name for p in d.iterdir()) == sorted(list(allm['E2']['files'].values()) + ['mo_hinh.json'])


@pytest.mark.parametrize('model', MODELS)
def test_saved_weights_reproduce_test_predictions_and_metrics(model):
    meta = _meta(model)['E2']
    assert meta['dataset_version'] == CFG['dataset_version']
    ds = _dataset(meta['H'], meta['max_gap'])
    v = ds.view('FINAL', meta['feats'], meta['L'], meta['fut'])
    saved = S.read_pred(CFG, 'test', meta['H'], 'chinh', model, 'E2')
    met = S.read_metrics(CFG, 'test', meta['H'], 'chinh', model, 'E2').set_index('seed')
    for seed, fname in meta['files'].items():
        got = _predict(meta, out_dir(CFG, model) / 'final_models' / fname, v)
        s = saved[saved.seed == int(seed)].pivot(index='ngay_goc', columns='h', values='Q_du_bao')
        assert (s.index.values == v.dates[v.idx_eval].values).all()
        assert np.allclose(got, s.values, rtol=1e-3, atol=0.05), (model, seed)
        nse = summary(v.QO[v.idx_eval], got, v.M[v.idx_eval])['NSE']
        assert abs(nse - met.loc[int(seed), 'NSE']) < 1e-3, (model, seed, nse, met.loc[int(seed), 'NSE'])
