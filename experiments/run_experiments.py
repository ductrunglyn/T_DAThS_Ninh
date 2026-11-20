"""Chạy toàn bộ hệ thống thực nghiệm theo đúng thứ tự (thực thi các notebook bằng nbclient).

    python experiments/run_experiments.py              # chạy đầy đủ 01 → 09 (~22 phút, RTX 5090), ghi đầu ra vào chính các notebook ở thư mục gốc
    python experiments/run_experiments.py --fast       # chạy thử nhanh: lưới nhỏ, kết quả ở results_fast/, notebook ở results_fast/_notebooks/
    python experiments/run_experiments.py --only 01_PhanTich_XuLy_DuLieu.ipynb 07_SoSanh.ipynb   # chỉ chạy các notebook này
"""
import argparse
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
ORDER = ['01_PhanTich_XuLy_DuLieu.ipynb', '02_Dataset_ChiaTap.ipynb', '03_Baseline.ipynb',
         '04_RNN.ipynb', '05_LSTM.ipynb', '06_GRU.ipynb', '07_SoSanh.ipynb', '08_DieuTra_LSTM_E5.ipynb',
         '09_KeHoachXa_HuaNa.ipynb']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    if a.fast:
        os.environ['NINH_FAST'] = '1'
    out_dir = ROOT / 'results_fast/_notebooks' if a.fast else ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in a.only or ORDER:
        t0 = time.time()
        nb = nbformat.read(ROOT / name, as_version=4)
        NotebookClient(nb, timeout=None, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
        nbformat.write(nb, out_dir / name)
        print(f'{name}: xong sau {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    sys.exit(main())
