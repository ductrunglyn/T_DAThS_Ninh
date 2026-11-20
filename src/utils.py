"""Tiện ích chung: đường dẫn gốc dự án, đọc cấu hình."""
import os
from pathlib import Path
import json
import yaml
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    p = Path(path) if path else ROOT / 'experiments' / 'config.yaml'
    with open(p, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    for k in ('start', 'end'):
        cfg['period'][k] = pd.Timestamp(cfg['period'][k])
    for fd in cfg['splits']['folds']:
        for k in ('train_end', 'val_start', 'val_end'):
            fd[k] = pd.Timestamp(fd[k])
    for k in ('train_end', 'test_start'):
        cfg['splits']['final'][k] = pd.Timestamp(cfg['splits']['final'][k])
    if os.environ.get('NINH_FAST'):          # chạy thử nhanh: lưới nhỏ, ít hạt giống, kết quả ở results_fast/
        cfg['paths']['results'] = 'results_fast'
        cfg['dataset_version'] = cfg['dataset_version'] + '_fast'
        for m in cfg['models'].values():
            if 'grid' in m:
                m['grid'] = {k: v[:1] for k, v in m['grid'].items()}
                m['grid']['L'] = [7]
        cfg['models']['extra_trees']['fixed']['n_estimators'] = 50
        cfg['models']['xgboost']['fixed']['max_estimators'] = 100
        cfg['seeds'] = {'tune': [0], 'cv': [0], 'final': [0, 1]}
        cfg['deep']['max_epochs'], cfg['deep']['patience'] = 6, 3
        cfg['weights']['settings'] = cfg['weights']['settings'][:2]
        cfg['penalty']['lambda'] = cfg['penalty']['lambda'][:1]
        cfg['bootstrap']['n'] = 200
    return cfg


def path(cfg, key, *parts):
    p = ROOT / cfg['paths'][key]
    for x in parts:
        p = p / x
    return p


def results_dir(cfg, *parts):
    p = path(cfg, 'results', *parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dataset_meta(cfg):
    """Thông tin dataset chuẩn (ngưỡng khoảng thiếu đã chọn...) do notebook 02 lưu ra."""
    f = path(cfg, 'results', 'dataset', 'dataset_meta.json')
    if f.exists():
        return json.loads(f.read_text(encoding='utf-8'))
    return {'max_gap': cfg['imputation']['default_gap']}
