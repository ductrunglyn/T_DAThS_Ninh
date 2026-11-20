"""Đóng băng bộ dữ liệu chuẩn (A5–A6) và kiểm tra trước khi dùng.

data/frozen/<version>/
  daily_clean.csv         bảng ngày đã làm sạch (NaN ở ngày thiếu)
  features_physical.csv   đặc trưng đơn vị vật lý sau điền nhân quả + cờ `unreliable`
  samples_H1.csv, samples_H10.csv   ngày gốc, cờ hợp lệ, vai trò trong từng fold / FINAL
  feature_sets.json       danh sách biến của từng tập
  data_dictionary_raw.csv, data_dictionary_features.csv
  split_report.csv        số mẫu, ngày đầu/cuối, Q max, Q90/Q95 train từng tách
  qc_flags.csv            ngày Q CĐ nghi nhiễu tính toán (CHỈ dùng cho phân tích độ nhạy khi chấm điểm)
  manifest.json           phiên bản, SHA-256 từng file, hash cấu hình và mã src/data
Bộ đã đóng băng không bị ghi đè: freeze() báo lỗi nếu nội dung mới khác bản đã có (phải tạo phiên bản mới).
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.utils import ROOT, path

CFG_KEYS = ['period', 'files', 'quality', 'imputation', 'horizons', 'splits', 'feature_groups', 'feature_sets']


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def config_hash(cfg, max_gap):
    sub = {k: cfg[k] for k in CFG_KEYS}
    sub['max_gap'] = max_gap
    return hashlib.sha256(json.dumps(sub, sort_keys=True, default=str).encode()).hexdigest()


def code_hash():
    h = hashlib.sha256()
    for p in sorted((ROOT / 'src' / 'data').glob('*.py')):
        if p.name in ('freeze.py', 'dictionary.py', 'hydro_analysis.py'):
            continue
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def frozen_dir(cfg, version=None):
    return ROOT / 'data' / 'frozen' / (version or cfg['dataset_version'])


def freeze(cfg, max_gap, suspect_dates=(), version=None):
    from src.data.windowing import UnifiedDataset, load_daily
    from src.data.feature_engineering import feature_list
    from src.data.dictionary import raw_dictionary, feature_dictionary
    out = frozen_dir(cfg, version)
    tmp = out.with_name(out.name + '_tmp')
    tmp.mkdir(parents=True, exist_ok=True)
    daily = load_daily(cfg, frozen=False)
    ds = {H: UnifiedDataset(cfg, H, max_gap, daily) for H in (cfg['horizons']['main'], cfg['horizons']['extended'])}
    d1 = ds[cfg['horizons']['main']]
    daily.to_csv(tmp / 'daily_clean.csv', encoding='utf-8-sig')
    d1.F.assign(unreliable=d1.filled['unreliable'].values).to_csv(tmp / 'features_physical.csv', encoding='utf-8-sig')
    for H, d in ds.items():
        d.sample_table().to_csv(tmp / f'samples_H{H}.csv', index=False, encoding='utf-8-sig')
    (tmp / 'feature_sets.json').write_text(json.dumps({s: feature_list(cfg, s) for s in cfg['feature_sets']},
                                                      ensure_ascii=False, indent=1), encoding='utf-8')
    raw_dictionary().to_csv(tmp / 'data_dictionary_raw.csv', encoding='utf-8-sig')
    feature_dictionary(cfg, max_gap).to_csv(tmp / 'data_dictionary_features.csv', encoding='utf-8-sig')
    pd.concat({f'H{H}': d.summary() for H, d in ds.items()}).to_csv(tmp / 'split_report.csv', encoding='utf-8-sig')
    pd.DataFrame({'date': pd.to_datetime(list(suspect_dates)), 'cờ': 'Q CĐ nghi nhiễu tính toán (răng cưa)'}) \
        .to_csv(tmp / 'qc_flags.csv', index=False, encoding='utf-8-sig')
    files = {p.name: {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in sorted(tmp.glob('*')) if p.name != 'manifest.json'}
    man = {'version': out.name, 'max_gap': max_gap, 'config_sha256': config_hash(cfg, max_gap), 'code_sha256': code_hash(),
           'period': [str(cfg['period']['start'].date()), str(cfg['period']['end'].date())],
           'samples': {f'H{H}': {'n': len(d.S), 'first_origin': str(d.S.date.iloc[0].date()),
                                 'last_origin': str(d.S.date.iloc[-1].date()), 'excluded_long_gap': int((~d.S.reliable).sum())}
                       for H, d in ds.items()},
           'files': files}
    if out.exists():
        old = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
        same = {k: v['sha256'] for k, v in old['files'].items()} == {k: v['sha256'] for k, v in files.items()}
        for p in tmp.glob('*'):
            p.unlink()
        tmp.rmdir()
        if not same:
            raise RuntimeError(f'{out} đã đóng băng và nội dung mới KHÁC — hãy tạo phiên bản mới (dataset_version)')
        return old, False
    man['created'] = datetime.now().isoformat(timespec='seconds')
    (tmp / 'manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')
    tmp.rename(out)
    return man, True


def verify(cfg, version=None):
    """Kiểm tra SHA-256 từng file, hash cấu hình và hash mã src/data so với manifest. Lỗi nếu không khớp."""
    d = frozen_dir(cfg, version)
    if not (d / 'manifest.json').exists():
        raise FileNotFoundError(f'chưa có bộ dữ liệu đóng băng {d} — chạy notebook 02 trước')
    man = json.loads((d / 'manifest.json').read_text(encoding='utf-8'))
    for name, info in man['files'].items():
        if sha256(d / name) != info['sha256']:
            raise RuntimeError(f'{name} đã bị thay đổi so với manifest {man["version"]}')
    if config_hash(cfg, man['max_gap']) != man['config_sha256']:
        raise RuntimeError('cấu hình dữ liệu trong config.yaml khác với lúc đóng băng')
    if code_hash() != man['code_sha256']:
        raise RuntimeError('mã src/data đã thay đổi so với lúc đóng băng')
    return man
