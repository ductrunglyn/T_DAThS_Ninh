"""Sinh các notebook mỏng 01–10 vào thư mục gốc dự án (cùng chỗ với src/, data/, results/).

    python experiments/build_notebooks.py                 # sinh lại cả 10 notebook (đầu ra đã chạy sẽ mất)
    python experiments/build_notebooks.py --only 01 07    # chỉ sinh lại notebook 01 và 07

Notebook 02–06, 08, 09 (và 10, không ghi đè kết quả chính) có bước huấn luyện; sinh lại rồi chạy lại chúng sẽ huấn luyện lại mô hình.
"""
import argparse
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell as md, new_code_cell as code

ROOT = Path(__file__).resolve().parents[1]
_ap = argparse.ArgumentParser()
_ap.add_argument('--only', nargs='*', help='tiền tố số của notebook cần sinh, ví dụ 01 07')
ONLY = _ap.parse_args().only
SETUP = r"""# ── Google Colab: gắn Google Drive, vào thư mục dự án, cài đúng phiên bản thư viện. Chạy trên máy thì tự bỏ qua. ──
import os, sys, glob, subprocess
DU_AN = '/content/drive/MyDrive/T_DAThS_Ninh'   # ← thư mục dự án trên Google Drive (notebook 01, mục 0 tạo ra)
if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive')
    if not os.path.isdir(os.path.join(DU_AN, 'src')):          # không thấy → tự tìm trong My Drive (sâu tối đa 3 cấp)
        hit = [h for k in range(1, 4) for h in glob.glob('/content/drive/MyDrive/' + '*/' * k + 'src/results_store.py')]
        assert hit, ('Chưa có dự án trên Google Drive: mở notebook 01_PhanTich_XuLy_DuLieu và chạy 2 ô của mục 0 '
                     '(lấy code từ GitHub, giải nén data/kết quả), hoặc sửa biến DU_AN ở đầu ô này')
        DU_AN = os.path.dirname(os.path.dirname(hit[0]))
    os.chdir(DU_AN)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements_colab.txt'], check=True)

import json, warnings
from pathlib import Path
ROOT = Path.cwd()
while not (ROOT / 'src' / 'results_store.py').exists():
    assert ROOT != ROOT.parent, 'Không tìm thấy thư mục dự án (thư mục chứa src/): mở notebook từ trong dự án'
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_colwidth', 80)
plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': .3})
from src.utils import load_config, results_dir, dataset_meta
cfg = load_config()
print('Thư mục dự án:', ROOT, '| kết quả lưu ở:', cfg['paths']['results'])
print('Thiết bị:', f'GPU {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else
      'CPU (trên Colab: Runtime → Change runtime type → GPU để chạy nhanh hơn)')"""


def save(name, cells):
    if ONLY and name[:2] not in ONLY:
        return
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["accelerator"] = "GPU"                         # Google Colab tự chọn runtime GPU khi mở notebook
    nb.metadata["colab"] = {"provenance": [], "toc_visible": True}
    nbformat.write(nb, ROOT / name)
    print('wrote', name, len(cells))


# =====================================================================================================================
# 01 — phân tích và xử lý dữ liệu
C = [md(r"""# 01. Phân tích và xử lý dữ liệu

Đọc dữ liệu gốc (`data/raw/`), làm sạch, hợp nhất thành **một bảng ngày**, kiểm tra chất lượng và phân tích quan hệ giữa các biến.
Notebook này **không** điền thiếu, không chuẩn hoá, không chia tập: bảng `data/processed/cuadat_huana_mua_daily.csv` giữ nguyên
ngày thiếu (NaN) cùng cờ thiếu; mọi bước tiền xử lý cho mô hình nằm trong `src/data/` và được dựng một lần ở notebook 02.

Mã nguồn: `src/data/quality_control.py`."""),
     md(r"""## 0. Chuẩn bị trên Google Colab (chạy trên máy thì bỏ qua mục này)

Code lấy từ GitHub, còn dữ liệu và kết quả lấy từ Google Drive. Cả hai được đặt vào **cùng một thư mục trên Drive**
(`MyDrive/T_DAThS_Ninh`), nên kết quả không mất khi tắt Colab và các notebook 02–10 dùng chung thư mục này.

1. Tải hai file nén lên **My Drive** (thư mục gốc):
   - `Ninh_data.zip` (bắt buộc): chứa thư mục `data/`;
   - `Ninh_results.zip` (không bắt buộc): chứa thư mục `results/` (kết quả và weight đã huấn luyện). Không có file này thì
     kết quả được sinh ra khi chạy notebook 03–09.
2. **Runtime → Change runtime type → GPU (T4)**.
3. Chạy ô 0a (lấy code), ô 0b (giải nén), rồi chạy tiếp các ô bên dưới như bình thường.

Chỉ cần làm mục này **một lần**. Các notebook khác chỉ cần mở và chạy: ô đầu tiên của chúng tự gắn Drive và tìm thư mục dự án."""),
     code(r"""# ── 0a. (Colab) Lấy code từ GitHub về Google Drive. Chạy trên máy thì tự bỏ qua. ──
import os, sys, subprocess
REPO  = 'https://github.com/ductrunglyn/T_DAThS_Ninh.git'
DU_AN = '/content/drive/MyDrive/T_DAThS_Ninh'          # thư mục dự án trên Drive: code + data/ + results/
if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive')
    if os.path.isdir(os.path.join(DU_AN, '.git')):      # đã clone trước đó → cập nhật code mới nhất
        r = subprocess.run(['git', '-C', DU_AN, 'pull', '--ff-only'], capture_output=True, text=True)
        print(r.stdout or r.stderr)
        if r.returncode:
            print('Không cập nhật được (có file đã sửa trên Drive): giữ nguyên code hiện tại.')
    elif os.path.isdir(os.path.join(DU_AN, 'src')):     # thư mục dự án chép tay lên Drive, không phải bản clone
        print('Đã có code ở', DU_AN, '(không phải bản clone): dùng luôn.')
    else:
        subprocess.run(['git', 'clone', REPO, DU_AN], check=True)
    os.chdir(DU_AN)
    print('Thư mục dự án:', DU_AN); print(sorted(os.listdir(DU_AN)))
else:
    print('Không chạy trên Colab: bỏ qua.')"""),
     code(r"""# ── 0b. (Colab) Giải nén data và kết quả (file .zip đã tải lên Drive) vào thư mục dự án. Chạy trên máy thì tự bỏ qua. ──
import zipfile
FILE_NEN = {'data':    '/content/drive/MyDrive/Ninh_data.zip',       # bắt buộc
            'results': '/content/drive/MyDrive/Ninh_results.zip'}    # không bắt buộc
GHI_DE = False      # True: giải nén lại dù thư mục đã có (ghi đè các file cùng tên)
if 'google.colab' in sys.modules:
    for ten, f in FILE_NEN.items():
        dich = os.path.join(DU_AN, ten)
        if os.path.isdir(dich) and os.listdir(dich) and not GHI_DE:
            print(f'{ten}/ đã có: bỏ qua (đặt GHI_DE = True để giải nén lại).')
        elif os.path.isfile(f):
            with zipfile.ZipFile(f) as z:
                ds = [n for n in z.namelist() if not n.startswith('__MACOSX')]
                ca_thu_muc = all(n.startswith(ten + '/') for n in ds)          # nén cả thư mục hay chỉ phần bên trong
                z.extractall(DU_AN if ca_thu_muc else dich, members=ds)
            print(f'Đã giải nén {f} → {dich}')
        elif ten == 'data':
            raise FileNotFoundError(f'Không thấy {f}: tải Ninh_data.zip lên My Drive, hoặc sửa đường dẫn trong FILE_NEN.')
        else:
            print(f'Không thấy {f}: bỏ qua. Kết quả sẽ được sinh ra khi chạy notebook 03–09.')
else:
    print('Không chạy trên Colab: bỏ qua.')"""),
     md(r"""## Thiết lập"""),
     code(SETUP + r"""
from src.data import quality_control as qc
OUT = results_dir(cfg, 'phan_tich_du_lieu')"""),
     md(r"""## 1. Đọc, làm sạch và hợp nhất

Giai đoạn chung của hai hồ 01/01/2019 – 12/08/2026 (bỏ 13/08/2026 vì số giờ phát điện bằng 0 ở cả hai hồ). Mưa CHIRPS trung bình
vùng: Vùng 1 — lưu vực Hủa Na; Vùng 2 — khu giữa Hủa Na và Cửa Đạt."""),
     code(r"""df, log = qc.build_daily(cfg)
for l in log: print('•', l)
f = ROOT / cfg['paths']['processed'] / 'cuadat_huana_mua_daily.csv'
f.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(f, encoding='utf-8-sig')
print(f'\nĐã lưu {f.relative_to(ROOT)}: {df.shape[0]} ngày × {df.shape[1]} cột; Cửa Đạt thiếu {df.cd_missing.sum()}, Hủa Na thiếu {df.hn_missing.sum()} ngày')
display(df.describe().T.round(2))"""),
     md(r"""## 2. Ngày thiếu theo độ dài khoảng thiếu

Phân loại phục vụ thiết kế điền thiếu nhân quả (notebook 02): 1 ngày, 2–3 ngày, 4–7 ngày, > 7 ngày."""),
     code(r"""gaps = qc.gap_table(df)
tab = gaps.groupby(['trạm', 'loại'])['số ngày'].agg(['count', 'sum']).rename(columns={'count': 'số đoạn', 'sum': 'số ngày'})
display(tab); gaps.to_csv(OUT / 'khoang_thieu.csv', index=False, encoding='utf-8-sig')
print('Khoảng thiếu > 3 ngày:'); display(gaps[gaps['số ngày'] > 3])
print('Số ngày thiếu theo năm:')
display(pd.DataFrame({'Cửa Đạt': df.cd_missing.groupby(df.index.year).sum(), 'Hủa Na': df.hn_missing.groupby(df.index.year).sum()}).T)"""),
     md(r"""## 3. Kiểm tra chất lượng dữ liệu

### 3.1 Tổng hợp
1. Khớp nguồn. 2. Căn thời gian mưa: tương quan mưa(t − k) với *mức tăng* log Q CĐ(t). 3. Lan truyền Hủa Na → Cửa Đạt ở thang
ngày. 4. Nhiễu "răng cưa" của Q đến CĐ. 5. Đoạn giá trị không đổi ≥ 5 ngày."""),
     code(r"""audit = qc.quality_audit(cfg, df)
display(audit); audit.to_csv(OUT / 'kiem_tra_chat_luong.csv', encoding='utf-8-sig')"""),
     md(r"""### 3.1b Giờ phát điện và các ô bị đặt thành NaN

Số giờ phát điện mâu thuẫn với Q qua nhà máy ở nhiều ngày → biến `hn_gio` có độ tin cậy thấp (chỉ dùng ở tập phụ F4).
Bảng thứ hai liệt kê **mọi** ô bị đặt NaN hoặc thiếu so với lịch ngày liên tục."""),
     code(r"""hrs = qc.operation_hours_check(cfg); display(hrs); hrs.to_csv(OUT / 'gio_van_hanh.csv', encoding='utf-8-sig')
nanr = qc.nan_replacements(cfg, df); nanr.to_csv(OUT / 'o_dat_nan.csv', index=False, encoding='utf-8-sig')
display(nanr.groupby(['nguồn', 'cột', 'lý do']).size().rename('số ô').to_frame())"""),
     md(r"""### 3.2 Giá trị lặp của Q đến Hủa Na

Không tự động coi đoạn lặp là lỗi. Trong bộ dữ liệu chuẩn có hai biến **nhân quả** (chỉ dùng dữ liệu ≤ t):
`qhn_rep_len` = số ngày liên tiếp tính đến t mà Q đến HN bằng giá trị ngày trước; `qhn_rep_flag` = 1 khi chuỗi lặp đã kéo dài
≥ 5 ngày. Không có thông tin phân loại chất lượng từ nguồn số liệu nên không tạo biến `qhn_data_quality`."""),
     code(r"""rep = qc.repeat_runs(df.hn_qden, cfg['quality']['repeat_min'])
print(f'{len(rep)} đoạn, {rep["số ngày"].sum()} ngày; theo tháng bắt đầu:', rep['tháng'].value_counts().sort_index().to_dict())
print('Giá trị lặp phổ biến:', rep['giá trị'].round(2).value_counts().head(8).to_dict())
display(rep.sort_values('số ngày', ascending=False).head(10)); rep.to_csv(OUT / 'doan_lap_q_den_hn.csv', index=False, encoding='utf-8-sig')"""),
     md(r"""### 3.3 Nhiễu "răng cưa" của Q đến Cửa Đạt — đối chiếu nguyên nhân

Ngày răng cưa: Q tăng (giảm) > 30 % rồi giảm (tăng) > 30 % ngay ngày sau. Mỗi ngày được đối chiếu với mưa hai vùng (t−2 … t),
thay đổi Q ra Hủa Na, thay đổi mực nước và Q xả Cửa Đạt. Ngày **không có mưa ≥ 10 mm và Q ra Hủa Na đổi < 35 %** được đánh dấu
"nghi nhiễu tính toán". Tên nhãn được giữ từ trước khi có xác nhận: đơn vị cung cấp đã xác nhận Q đến hồ là số liệu đo
(13/11/2026, VX-02), nên trong báo cáo các ngày này được gọi là "răng cưa chưa rõ nguyên nhân".
**Không sửa và không làm trơn nhãn**: chỉ báo cáo. Đầu vào có thêm biến trung bình trượt mũ nhân quả `qcd_ema3`, được kiểm tra
như một thực nghiệm phụ (P_EMA)."""),
     code(r"""saw = qc.sawtooth_table(df, cfg['quality']['sawtooth_jump'])
saw.to_csv(OUT / 'rang_cua_q_cd.csv', index=False, encoding='utf-8-sig')
print(saw['đánh giá'].value_counts().to_string())
print('\nTheo hướng:'); display(pd.crosstab(saw['hướng'], saw['đánh giá']))
print('Theo tháng:'); display(pd.crosstab(pd.to_datetime(saw['ngày']).dt.month, saw['đánh giá']).T)
g = saw.groupby('đánh giá')[['Q(t)', 'ΔZ CĐ (t) cm', 'ΔZ CĐ (t+1) cm', 'ΔQ xả CĐ (t)']].agg(lambda s: s.abs().median())
print('Trung vị |giá trị| theo nhóm:'); display(g.round(1))
display(saw.head(8))"""),
     md(r"""### 3.4 Đồng bộ thời gian giữa mưa CHIRPS và số liệu vận hành

Từ các file dữ liệu **không xác định được** quy ước ngày của từng nguồn: CHIRPS là tổng mưa theo ngày lịch của sản phẩm, còn số
liệu vận hành hồ là trung bình ngày theo quy ước báo cáo của đơn vị vận hành (có thể 0 h – 24 h hoặc 7 h – 7 h giờ địa phương).
Đơn vị cung cấp đã xác nhận cả hai nguồn đều theo ngày lịch (13/11/2026, VX-01, VX-06). Kiểm tra thực nghiệm dưới đây chỉ cho biết **không có độ lệch nguyên ngày** rõ ràng; nó
không loại trừ được độ lệch dưới 1 ngày (ví dụ 7 giờ)."""),
     code(r"""dq = np.log1p(df.q_cd).diff(); ok = ~df.cd_missing & ~df.cd_missing.shift(1, fill_value=True)
rows = []
for v in ['mua_v1', 'mua_v2']:
    cc = qc.lag_corr(df[v], dq, ok, range(-3, 4))
    rows.append({'biến': v, **{f'k = {k}': c for k, c in cc.items()}})
lagr = pd.DataFrame(rows).set_index('biến'); display(lagr.round(3))
# theo trận mưa lớn: ngày Q bắt đầu tăng so với ngày mưa lớn nhất
ev = []
for v in ['mua_v1', 'mua_v2']:
    big = df[v][(df[v] >= 40)].index
    for d in big:
        w = df.q_cd.loc[d - pd.Timedelta(days=1):d + pd.Timedelta(days=3)]
        if w.isna().any() or len(w) < 5: continue
        ev.append({'vùng': v, 'ngày mưa': d.date(), 'mưa': df.loc[d, v], 'ngày Q tăng mạnh nhất (so với ngày mưa)': int((w.diff().idxmax() - d).days)})
ev = pd.DataFrame(ev)
print('Các ngày mưa ≥ 40 mm: ngày có mức tăng Q lớn nhất trong (−1 … +3):')
display(ev.groupby('vùng')['ngày Q tăng mạnh nhất (so với ngày mưa)'].value_counts().unstack(fill_value=0))
lagr.to_csv(OUT / 'dong_bo_thoi_gian_mua.csv', encoding='utf-8-sig')"""),
     md(r"""## 4. Phân tích quan hệ giữa các biến

Để không "nhìn" trước giai đoạn kiểm tra, tương quan và độ trễ chỉ tính trên **2019–2023** (ngày có quan trắc thật).

### 4.1 Chuỗi thời gian theo giai đoạn"""),
     code(r"""P1_END, P2_END, P3_START = pd.Timestamp('2023-12-31'), pd.Timestamp('2024-12-31'), pd.Timestamp('2025-01-01')
PER = pd.Series(np.where(df.index <= P1_END, '2019–2023', np.where(df.index <= P2_END, '2024', '2025–2026')), index=df.index)
fig, axs = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
axs[0].plot(df.q_cd, lw=.7); axs[0].set_ylabel('Q đến Cửa Đạt\n(m³/s)')
axs[1].plot(df.hn_qden, lw=.7, color='C7', label='Q đến Hủa Na'); axs[1].plot(df.hn_qxa, lw=.7, color='C1', label='Q ra Hủa Na')
axs[1].set_ylabel('Hủa Na\n(m³/s)'); axs[1].legend(fontsize=8, loc='upper left')
axs[2].plot(df.hn_mn, lw=.7, color='C5'); axs[2].set_ylabel('Mực nước\nHủa Na (m)')
axs[3].bar(df.index, df.mua_v1, width=1, color='C2', alpha=.7, label='Vùng 1 (lưu vực Hủa Na)')
axs[3].bar(df.index, -df.mua_v2, width=1, color='C9', alpha=.7, label='Vùng 2 (giữa 2 hồ), vẽ âm')
axs[3].set_ylabel('Mưa\n(mm/ngày)'); axs[3].legend(fontsize=8, loc='lower left')
for ax in axs:
    ax.axvspan(pd.Timestamp('2024-01-01'), P2_END, color='C4', alpha=.08); ax.axvspan(P3_START, df.index[-1], color='C3', alpha=.08)
plt.tight_layout(); plt.savefig(OUT / 'fig01_chuoi_thoi_gian.png'); plt.show()
real = ~df.cd_missing
st = df[real].groupby(PER[real])['q_cd'].describe(percentiles=[.5, .95, .99]).loc[['2019–2023', '2024', '2025–2026']]
display(st.round(1)); st.to_csv(OUT / 'phan_bo_q_theo_giai_doan.csv', encoding='utf-8-sig')"""),
     md(r"""### 4.2 Tương quan (2019–2023, thang log)"""),
     code(r"""an = df.copy()
for v in ['v1', 'v2']: an[f'mua_{v}_7'] = an[f'mua_{v}'].rolling(7, min_periods=7).sum()
COR = {'q_cd': 'Q đến CĐ', 'hn_qden': 'Q đến HN', 'hn_qxa': 'Q ra HN', 'hn_qnm': 'Q nhà máy HN', 'hn_qtran': 'Q tràn HN',
       'hn_mn': 'Mực nước HN', 'mua_v1': 'Mưa V1', 'mua_v2': 'Mưa V2', 'mua_v1_7': 'Mưa V1 7 ngày', 'mua_v2_7': 'Mưa V2 7 ngày'}
trd = an.loc[(an.index <= P1_END) & ~df.cd_missing & ~df.hn_missing, list(COR)].dropna()
lg = [c for c in COR if c != 'hn_mn']; trd[lg] = np.log1p(trd[lg])
cor = trd.corr().rename(index=COR, columns=COR); display(cor.round(2)); cor.to_csv(OUT / 'tuong_quan_bien.csv', encoding='utf-8-sig')
fig, ax = plt.subplots(figsize=(8, 6.5)); im = ax.imshow(cor.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(cor))); ax.set_xticklabels(cor.columns, rotation=60, ha='right', fontsize=8)
ax.set_yticks(range(len(cor))); ax.set_yticklabels(cor.index, fontsize=8)
for i in range(len(cor)):
    for j in range(len(cor)): ax.text(j, i, f'{cor.values[i, j]:.2f}', ha='center', va='center', fontsize=7, color='w' if abs(cor.values[i, j]) > .6 else 'k')
plt.colorbar(im, ax=ax, shrink=.8); ax.grid(False); plt.tight_layout(); plt.savefig(OUT / 'fig02_tuong_quan.png'); plt.show()"""),
     md(r"""### 4.3 Mực nước hồ Hủa Na và dòng chảy ra"""),
     code(r"""rd = df.loc[~df.hn_missing].copy()
rd['nhóm mực nước (m)'] = pd.cut(rd.hn_mn, [0, 225, 230, 235, 238, 241], labels=['< 225', '225–230', '230–235', '235–238', '≥ 238'], right=False)
lv = rd.groupby('nhóm mực nước (m)', observed=True).agg(**{'số ngày': ('hn_mn', 'size'), 'Q đến TB': ('hn_qden', 'mean'),
     'Q ra TB': ('hn_qxa', 'mean'), 'Q tràn TB': ('hn_qtran', 'mean'), '% ngày xả tràn > 50': ('hn_qtran', lambda s: 100 * (s > 50).mean())})
lv['Q ra / Q đến'] = lv['Q ra TB'] / lv['Q đến TB']; display(lv.round(2)); lv.to_csv(OUT / 'muc_nuoc_huana.csv', encoding='utf-8-sig')
fig, ax = plt.subplots(figsize=(7, 4)); sc = ax.scatter(rd.hn_mn, rd.hn_qxa, c=np.log1p(rd.hn_qden), s=5, cmap='viridis')
ax.set_yscale('log'); ax.set_xlabel('Mực nước Hủa Na (m)'); ax.set_ylabel('Q ra Hủa Na (m³/s)'); plt.colorbar(sc, ax=ax, label='log(1 + Q đến HN)')
plt.tight_layout(); plt.savefig(OUT / 'fig03_muc_nuoc_huana.png'); plt.show()"""),
     md(r"""### 4.4 Độ trễ (2019–2023)

Tương quan giữa X(t − k) và Q đến CĐ(t), k = 0 … 10; tương quan riêng phần của Q ra Hủa Na sau khi loại ảnh hưởng của mưa hai vùng.
Đề cương mô tả độ trễ của Q và mưa trong phạm vi vài ngày; độ trễ cửa sổ L ∈ {3, 7, 14, 30} được chọn bằng CV ở các notebook mô hình."""),
     code(r"""tr = an.loc[:P1_END]; y = np.log1p(tr.q_cd)
Zc = pd.concat([np.log1p(tr[f'mua_{v}']).shift(j) for v in ['v1', 'v2'] for j in range(4)] + [np.log1p(tr.mua_v1_7), np.log1p(tr.mua_v2_7)], axis=1)
def resid(v, Z):
    Z1 = np.c_[np.ones(len(Z)), Z]; return v - Z1 @ np.linalg.lstsq(Z1, v, rcond=None)[0]
SER = {'Q ra HN': np.log1p(tr.hn_qxa), 'Q đến HN': np.log1p(tr.hn_qden), 'Mực nước HN': tr.hn_mn, 'Mưa V1': np.log1p(tr.mua_v1), 'Mưa V2': np.log1p(tr.mua_v2)}
rows = []
for k in range(11):
    ok = (~tr.cd_missing) & ~tr.hn_missing.shift(k, fill_value=True).astype(bool) & Zc.notna().all(axis=1)
    row = {'trễ k': k, 'n': int(ok.sum()), **{f'r {nm}': np.corrcoef(s.shift(k)[ok], y[ok])[0, 1] for nm, s in SER.items()}}
    xx, yy, ZZ = SER['Q ra HN'].shift(k)[ok].values, y[ok].values, Zc[ok].values
    row['r riêng phần Q ra HN'] = np.corrcoef(resid(xx, ZZ), resid(yy, ZZ))[0, 1]; rows.append(row)
lag = pd.DataFrame(rows).set_index('trễ k'); display(lag.round(3)); lag.to_csv(OUT / 'do_tre_tuong_quan.csv', encoding='utf-8-sig')
fig, ax = plt.subplots(figsize=(9, 3.8))
for c, mk in zip(lag.columns[1:], 'osd^vx'): ax.plot(lag.index, lag[c], marker=mk, label=c.replace('r ', ''))
ax.set_xlabel('độ trễ k (ngày)'); ax.set_ylabel('tương quan với Q đến CĐ(t)'); ax.legend(fontsize=7, ncol=2); ax.set_xticks(range(11))
plt.tight_layout(); plt.savefig(OUT / 'fig04_do_tre_tuong_quan.png'); plt.show()"""),
     md(r"""### 4.5 Quan hệ theo mùa và theo trạng thái dòng chảy (2019–2023)

Tương quan giữa X(t − k) và log Q CĐ(t), k = 0 … 3, tách theo mùa (lũ T7–T10 / kiệt) và theo trạng thái dòng chảy Cửa Đạt
(thấp < Q50, trung bình Q50–Q90, cao ≥ Q90 của 2019–2023). Bảng thứ hai: tương quan giữa **biến động** Δ log Q ra HN(t − k) và
Δ log Q CĐ(t) theo mùa. Đây là tương quan trên số liệu trung bình ngày — **không** xác định được chính xác thời gian truyền lũ
(nhỏ hơn 1 ngày không phân giải được); tương quan đồng thời (k = 0) có thể phản ánh cả mưa chung trên hai lưu vực."""),
     code(r"""from src.data import hydro_analysis as ha
st_tab, (q50, q90) = ha.by_season_state(df, P1_END)
display(st_tab.round(3)); st_tab.to_csv(OUT / 'tuong_quan_theo_mua_trang_thai.csv', encoding='utf-8-sig')
dl = ha.delta_relation(df, P1_END); print('Δ log Q ra HN(t − k) với Δ log Q CĐ(t):'); display(dl.round(3)); dl.to_csv(OUT / 'bien_dong_q_ra_hn.csv', encoding='utf-8-sig')
fig, axs = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
for ax, v in zip(axs, ['Q đến HN', 'Q ra HN', 'Mưa V2']):
    t_ = st_tab.xs(v, level='biến')[['k = 0', 'k = 1', 'k = 2', 'k = 3']]
    for g, row in t_.iterrows(): ax.plot(range(4), row.values, marker='o', label=g)
    ax.set_title(v); ax.set_xlabel('độ trễ k (ngày)'); ax.set_xticks(range(4))
axs[0].set_ylabel('tương quan với log Q CĐ(t)'); axs[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / 'fig05_tuong_quan_mua_trang_thai.png'); plt.show()"""),
     md(r"""### 4.6 Mưa và dòng chảy theo mùa lũ (tháng 7–10)"""),
     code(r"""fl = df[df.index.month.isin([7, 8, 9, 10])]; g = fl.groupby(fl.index.year)
yr = pd.DataFrame({'mưa V1 T7–T10 (mm)': g.mua_v1.sum(), 'mưa V1 ngày max': g.mua_v1.max(), 'mưa V2 T7–T10 (mm)': g.mua_v2.sum(),
                   'mưa V2 ngày max': g.mua_v2.max(), 'Q đến CĐ max': g.q_cd.max(), 'ngày thiếu Q CĐ': g.cd_missing.sum(), 'Q ra HN max': g.hn_qxa.max()}).round(0)
display(yr); yr.to_csv(OUT / 'mua_lu_theo_nam.csv', encoding='utf-8-sig')"""),
     md(r"""### 4.7 Thống kê mô tả theo mùa (chỉ ngày có quan trắc)

Đặc điểm dòng chảy Cửa Đạt, dòng chảy đến và xả Hủa Na, mực nước Hủa Na và mưa vùng theo mùa lũ (T7–T10) / mùa kiệt (T11–T6).
Giai đoạn phân tích quan hệ 2019–2023 (như mục 4.1–4.5) và cả chuỗi 2019–08/2026 (chỉ để mô tả; không dùng cho bất kỳ lựa
chọn nào của mô hình). "Q ra − Q đến HN" > 0: hồ Hủa Na xả nhiều hơn lượng nước đến (tháo nước hồ); < 0: hồ tích nước."""),
     code(r"""V = {'Q đến CĐ (m³/s)': ('q_cd', 'cd_missing'), 'Q đến HN (m³/s)': ('hn_qden', 'hn_missing'), 'Q ra HN (m³/s)': ('hn_qxa', 'hn_missing'),
     'Q ra − Q đến HN (m³/s)': ('hn_bal_raw', 'hn_missing'), 'Z HN (m)': ('hn_mn', 'hn_missing'), 'Mưa V1 (mm/ngày)': ('mua_v1', None), 'Mưa V2 (mm/ngày)': ('mua_v2', None)}
d_ = df.assign(hn_bal_raw=df.hn_qxa - df.hn_qden)
rows = []
for per, end in [('2019–2023', P1_END), ('2019–08/2026', df.index[-1])]:
    for mua, months in [('mùa lũ T7–T10', [7, 8, 9, 10]), ('mùa kiệt T11–T6', [11, 12, 1, 2, 3, 4, 5, 6])]:
        sub = d_.loc[:end]; sub = sub[sub.index.month.isin(months)]
        for nm, (c, miss) in V.items():
            x = sub[c][~sub[miss]] if miss else sub[c]
            x = x.dropna()
            rows.append({'giai đoạn': per, 'mùa': mua, 'biến': nm, 'n': len(x), 'TB': x.mean(), 'ĐLC': x.std(), 'min': x.min(),
                         'Q50': x.quantile(.5), 'Q95': x.quantile(.95), 'max': x.max()})
mt = pd.DataFrame(rows).set_index(['giai đoạn', 'mùa', 'biến'])
display(mt.round(2)); mt.to_csv(OUT / 'thong_ke_mo_ta_theo_mua.csv', encoding='utf-8-sig')"""),
     md(r"""### 4.8 Kiểm tra cân bằng nước giữa hai hồ (chỉ mô tả)

Cửa Đạt nằm hạ lưu Hủa Na nên, về trung bình dài hạn, Q đến Cửa Đạt phải lớn hơn Q ra Hủa Na một lượng bằng dòng chảy gia nhập
khu giữa (V2). Bảng tính trung bình theo năm và mùa trên **cùng các ngày** có số liệu của cả hai hồ: "khu giữa" = Q đến CĐ − Q ra HN.
Giá trị âm kéo dài là dấu hiệu bất nhất giữa hai nguồn số liệu. Đơn vị cung cấp xác nhận cả hai chuỗi là số liệu đo (13/11/2026),
nên **nguyên nhân chưa giải thích được** (có thể Q ra HN thiếu thành phần xả khác, VX-04); dữ liệu và nhãn được giữ nguyên."""),
     code(r"""c = df[~df.cd_missing & ~df.hn_missing].copy(); c['khu giữa'] = c.q_cd - c.hn_qxa
c['năm'] = c.index.year; c['mùa'] = np.where(c.index.month.isin([7, 8, 9, 10]), 'lũ T7–T10', 'kiệt T11–T6')
cb = c.groupby(['năm', 'mùa']).agg(**{'số ngày': ('q_cd', 'size'), 'Q đến CĐ TB': ('q_cd', 'mean'), 'Q ra HN TB': ('hn_qxa', 'mean'),
     'Q đến HN TB': ('hn_qden', 'mean'), 'khu giữa TB (m³/s)': ('khu giữa', 'mean'), '% ngày Q đến CĐ < Q ra HN': ('khu giữa', lambda x: 100 * (x < 0).mean()),
     'mưa V2 TB (mm/ngày)': ('mua_v2', 'mean')})
cb.loc[('cả chuỗi', ''), :] = [len(c), c.q_cd.mean(), c.hn_qxa.mean(), c.hn_qden.mean(), c['khu giữa'].mean(), 100 * (c['khu giữa'] < 0).mean(), c.mua_v2.mean()]
display(cb.round(1)); cb.to_csv(OUT / 'can_bang_nuoc_khu_giua.csv', encoding='utf-8-sig')""")]
save('01_PhanTich_XuLy_DuLieu.ipynb', C)

# =====================================================================================================================
# 02 — dataset chuẩn
C = [md(r"""# 02. Bộ dữ liệu chuẩn và cách chia tập (dùng chung cho mọi mô hình)

Dựng **một** bộ dữ liệu chuẩn từ bảng ngày của notebook 01 (`src/data/`): điền thiếu nhân quả → đặc trưng → biến đổi log →
mẫu cửa sổ → vai trò train/val/test của từng fold. Chuẩn hoá z-score và ngưỡng Q90/Q95 được học **từ train của từng fold** khi
mô hình lấy dữ liệu (`UnifiedDataset.view`). Mọi mô hình — Persistence, Ridge-ARX, Extra Trees, XGBoost, RNN, LSTM, GRU — đọc
cùng bộ dữ liệu này, cùng danh sách mẫu, cùng ngày đích.

| Thành phần | Nội dung |
|---|---|
| `X_seq` | N × L × F (RNN / LSTM / GRU) |
| `X_flat` | N × (L·F) (Ridge-ARX / Extra Trees / XGBoost); Persistence dùng Q CĐ tại ngày gốc |
| `Y`, `M` | N × H đích log(1+Q) chuẩn hoá; mặt nạ ô đích có quan trắc thật |
| ngày gốc, vai trò | `results/dataset/samples_H1.csv`, `samples_H10.csv` |
| metadata | `results/dataset/features_meta.csv`, `dataset_meta.json` |"""),
     code(SETUP + r"""
from src.data.windowing import UnifiedDataset, load_daily
from src.data.feature_engineering import metadata_table, feature_list
from src.data.splitting import fold_names
from src import pipeline
OUT = results_dir(cfg, 'dataset')
daily = load_daily(cfg)"""),
     md(r"""## 1. Điền thiếu nhân quả và chọn ngưỡng khoảng thiếu G

| Trường hợp | Xử lý |
|---|---|
| Khoảng thiếu ≤ G ngày | Forward fill (giá trị quan trắc gần nhất trước đó) + cờ thiếu `*_miss` + tuổi số liệu `*_age` |
| Khoảng thiếu > G ngày | Không điền làm đầu vào: **loại mẫu** có cửa sổ 30 ngày chạm vào khoảng này |
| Q CĐ thiếu ở ngày đích | Không tạo nhãn giả; ô đích bị loại khỏi hàm mất mát và chấm điểm |

G ∈ {1, 3, 7} được chọn **một lần cho mọi mô hình** bằng CV 3 fold với LSTM và GRU (L = 14, hidden 64, 1 lớp, tập biến F1, bài
toán t+1). Để so sánh công bằng, mọi G được chấm trên **cùng** các mẫu val hợp lệ với cả ba ngưỡng (giao)."""),
     code(r"""gap_tab, G, gmean = pipeline.select_gap(cfg)
display(gap_tab.round(3)); gap_tab.to_csv(OUT / 'chon_nguong_khoang_thieu.csv', index=False, encoding='utf-8-sig')
print('NSE CV trung bình (LSTM, GRU) theo G:', gmean.round(4).to_dict(), '→ chọn G =', G)"""),
     md(r"""## 2. Đặc trưng và các tập biến"""),
     code(r"""ds = {H: UnifiedDataset(cfg, H, G, daily) for H in (cfg['horizons']['main'], cfg['horizons']['extended'])}
d1 = ds[cfg['horizons']['main']]
meta = metadata_table(); display(meta); meta.to_csv(OUT / 'features_meta.csv', encoding='utf-8-sig')
fs = pd.DataFrame([{'tập biến': k, 'số biến': len(feature_list(cfg, k)), 'biến': ', '.join(feature_list(cfg, k))} for k in cfg['feature_sets']]).set_index('tập biến')
display(fs)"""),
     md(r"""## 3. Mẫu và chia tập

| Tách | Train (ngày đích ≤) | Đánh giá |
|---|---|---|
| F1 | 31/12/2020 | val 2021 |
| F2 | 31/12/2021 | val 2022 (đỉnh 1.436 m³/s) |
| F3 | 31/12/2023 | val 2024 (đỉnh 2.802 m³/s) |
| FINAL | 31/12/2024 | test 01/2025 – 08/2026 (**khoá**: không dùng để chọn siêu tham số, kiến trúc, tập biến hay trọng số) |

Đầu vào của mẫu val/test được phép dùng quan trắc thuộc giai đoạn trước (như khi dự báo thực tế); chỉ ngày **đích** của train và
val/test được tách rời. Năm 2023 không làm fold val vì Cửa Đạt thiếu trọn mùa lũ (02/08 – 16/10/2023)."""),
     code(r"""for H, d in ds.items():
    print(f'— H = {H}: {len(d.S)} ngày gốc ({d.S.date.iloc[0].date()} → {d.S.date.iloc[-1].date()}); '
          f'loại do chạm khoảng thiếu > {G} ngày: {(~d.S.reliable).sum()}')
    display(d.summary().round(1))
    d.sample_table().to_csv(OUT / f'samples_H{H}.csv', index=False, encoding='utf-8-sig')
fig, ax = plt.subplots(figsize=(12, 2.6))
for i, sp in enumerate(fold_names(cfg) + ['FINAL']):
    r = d1.S[sp]
    for role, c in [('train', 'C0'), ('val', 'C1'), ('test', 'C3')]:
        x = d1.S.date[r == role]; ax.scatter(x, np.full(len(x), i), s=2, color=c, label=role if i == 0 or (role == 'test' and sp == 'FINAL') else None)
ax.set_yticks(range(4)); ax.set_yticklabels(fold_names(cfg) + ['FINAL']); ax.legend(markerscale=5, fontsize=8, loc='lower left')
ax.set_title('Vai trò của từng ngày gốc (t+1)'); plt.tight_layout(); plt.savefig(OUT / 'fig_chia_tap.png'); plt.show()"""),
     md(r"""## 4. Kiểm tra thiết kế (tự động)"""),
     code(r"""checks = []
def check(name, cond):
    assert cond, name; checks.append(name)
f = d1.filled; obs = daily
for c in ['q_cd', 'hn_qxa', 'hn_qden', 'hn_mn']:
    check(f'điền nhân quả {c} = quan trắc gần nhất trước đó', np.allclose(obs[c].ffill(), f[c]))
check('tuổi số liệu chỉ dùng quá khứ', all(f.cd_age.iloc[t] == (0 if not obs.cd_missing.iloc[t] else f.cd_age.iloc[t - 1] + 1) for t in range(1, len(f))))
from src.data.quality_control import repeat_length_causal
x = obs.hn_qden.where(~obs.hn_missing).values
check('cờ lặp Q đến HN chỉ dùng dữ liệu ≤ t', np.array_equal(repeat_length_causal(x[:1000]), repeat_length_causal(x)[:1000]))
check('tổng mưa trượt chỉ dùng quá khứ', np.isclose(d1.F.mua_v1_7.iloc[500], obs.mua_v1.iloc[494:501].sum()))
for H, d in ds.items():
    tg = lambda ix: np.unique((d.S.date.values[ix][:, None] + np.arange(1, H + 1) * np.timedelta64(1, 'D')).ravel())
    for sp in fold_names(cfg) + ['FINAL']:
        v = d.view(sp, feature_list(cfg, 'F3'), 30)
        tr, ev = v.idx_train, v.idx_eval
        check(f'[H{H}/{sp}] ngày đích train và đánh giá không trùng', np.intersect1d(tg(tr), tg(ev)).size == 0)
        end = cfg['splits']['final']['train_end'] if sp == 'FINAL' else next(x for x in cfg['splits']['folds'] if x['name'] == sp)['train_end']
        check(f'[H{H}/{sp}] ngày đích train ≤ hết giai đoạn train', (d.S.date.values[tr] + np.timedelta64(H, 'D') <= np.datetime64(end)).all())
        td = d.train_days(sp); fz = feature_list(cfg, 'F3')
        mu, sd = d.T.loc[td, fz].mean(), d.T.loc[td, fz].std()
        bi = [c for c in fz if meta.loc[c, 'biến đổi'] == 'binary']; mu[bi], sd[bi] = 0.0, 1.0
        t_ = d.S.pos.iloc[ev[0]]
        check(f'[H{H}/{sp}] chuẩn hoá dùng μ, σ của ngày train (không chạm ngày đánh giá)',
              np.allclose(v.X_seq[ev[0], -1], ((d.T[fz].iloc[t_] - mu) / sd).values, atol=1e-4) and td[d.index > end].sum() == 0)
        check(f'[H{H}/{sp}] Q95 lấy từ train', np.isclose(v.q95, np.quantile(daily.q_cd[(daily.index <= end) & ~daily.cd_missing], .95)))
        check(f'[H{H}/{sp}] mẫu train/đánh giá không chạm khoảng thiếu dài', d.S.reliable.values[tr].all() and d.S.reliable.values[ev].all())
    i = 700; t = d.S.pos.iloc[i]; v = d.view('FINAL', feature_list(cfg, 'F3'), 14)
    sc_mu, sc_sd = d.T.loc[d.train_days('FINAL')].mean(), d.T.loc[d.train_days('FINAL')].std()
    check(f'[H{H}] bước cuối của cửa sổ là ngày gốc t', np.allclose(v.X_seq[i, -1, 0], (d.T.q_cd.iloc[t] - sc_mu.q_cd) / sc_sd.q_cd, atol=1e-5))
    check(f'[H{H}] đích = Q quan trắc t+1 … t+H', np.allclose(np.nan_to_num(v.QO[i]), np.nan_to_num(daily.q_cd.values[t + 1:t + H + 1])))
    check(f'[H{H}] test bắt đầu 01/01/2025 và không có ô đích thiếu', d.S.date[v.idx_eval].min() == pd.Timestamp('2025-01-01') and v.M[v.idx_eval].all())
check('mọi mô hình dùng cùng tập mẫu: X_flat là X_seq trải phẳng', np.allclose(v.X_flat[5], v.X_seq[5].ravel()))
for c in dict.fromkeys(checks): print('✔', c)
print(len(checks), 'phép kiểm tra đều đạt.')
(OUT / 'kiem_tra_thiet_ke.txt').write_text('\n'.join('✔ ' + c for c in dict.fromkeys(checks)) + f'\n{len(checks)} phép kiểm tra đều đạt.\n', encoding='utf-8')"""),
     md(r"""## 5. Từ điển dữ liệu, đóng băng phiên bản và kiểm tra toàn vẹn

Bộ dữ liệu được **đóng băng** ở `data/frozen/v1/` (bảng ngày, đặc trưng, mẫu H1/H10, từ điển dữ liệu, báo cáo chia tập, cờ chất
lượng, `manifest.json` với SHA-256 từng file, hash cấu hình và hash mã `src/data`). Mọi notebook mô hình gọi
`pipeline.get_dataset()`: kiểm tra manifest và kiểm tra tập mẫu dựng lại trùng bản đóng băng trước khi huấn luyện. Bản đã đóng
băng không bị ghi đè — thay đổi dữ liệu phải tạo phiên bản mới (`dataset_version`)."""),
     code(r"""from src.data.dictionary import raw_dictionary, feature_dictionary
from src.data.freeze import freeze, verify
display(raw_dictionary())
saw = pd.read_csv(results_dir(cfg, 'phan_tich_du_lieu') / 'rang_cua_q_cd.csv')
suspect = pd.to_datetime(saw.loc[saw['đánh giá'].str.startswith('nghi'), 'ngày'])
man, created = freeze(cfg, G, suspect)
print('Tạo mới' if created else 'Đã có và trùng khớp —', 'phiên bản', man['version'], '| G =', man['max_gap'])
print(json.dumps({k: v for k, v in man.items() if k != 'files'}, ensure_ascii=False, indent=1))
display(pd.DataFrame(man['files']).T)
verify(cfg); print('✔ manifest khớp')"""),
     code(r"""import subprocess
r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/'], capture_output=True, text=True, cwd=ROOT)
(OUT / 'kiem_tra_toan_ven_pytest.txt').write_text(r.stdout + r.stderr, encoding='utf-8')
print(r.stdout[-1500:]); assert r.returncode == 0, 'unit test thất bại'"""),
     md(r"""## 6. Đỉnh lũ trong tập test và lưu metadata"""),
     code(r"""peaks, thr = pipeline.test_peaks(cfg, d1)
print(f'Ngưỡng đỉnh = Q99 train cuối = {thr:.0f} m³/s; các đỉnh test:', [(p.date(), daily.q_cd[p]) for p in peaks])
meta_js = {'max_gap': G, 'gap_selection_nse_cv': {int(k): float(x) for k, x in gmean.items()},
           'n_samples': {H: len(d.S) for H, d in ds.items()}, 'test_peaks': [str(p.date()) for p in peaks], 'peak_threshold': thr}
(OUT / 'dataset_meta.json').write_text(json.dumps(meta_js, ensure_ascii=False, indent=1), encoding='utf-8')
print(meta_js)""")]
save('02_Dataset_ChiaTap.ipynb', C)

# =====================================================================================================================
MODEL_INTRO = {
    'rnn': r"""**RNN** cơ bản (Elman): $h_\tau = \tanh(W x_\tau + U h_{\tau-1} + b)$ — mô hình học sâu tham chiếu (baseline học sâu).""",
    'lstm': r"""**LSTM** (Hochreiter & Schmidhuber, 1997) — mô hình chính:
$$i_\tau, f_\tau, o_\tau = \sigma(W_{i,f,o} x_\tau + U_{i,f,o} h_{\tau-1} + b_{i,f,o}),\quad \tilde c_\tau = \tanh(W_c x_\tau + U_c h_{\tau-1} + b_c)$$
$$c_\tau = f_\tau \odot c_{\tau-1} + i_\tau \odot \tilde c_\tau,\qquad h_\tau = o_\tau \odot \tanh(c_\tau)$$""",
    'gru': r"""**GRU** (Cho và cs., 2014) — mô hình chính:
$$z_\tau = \sigma(W_z x_\tau + U_z h_{\tau-1} + b_z),\quad r_\tau = \sigma(W_r x_\tau + U_r h_{\tau-1} + b_r)$$
$$\tilde h_\tau = \tanh(W_h x_\tau + U_h (r_\tau \odot h_{\tau-1}) + b_h),\qquad h_\tau = (1 - z_\tau) \odot h_{\tau-1} + z_\tau \odot \tilde h_\tau$$""",
}
PROTOCOL = r"""**Quy trình (giống hệt cho mọi mô hình, mã ở `src/pipeline.py`)**
1. Dò siêu tham số trên tập biến F1 (E2), hàm mất mát cơ bản, bài toán t+1, CV 3 fold; tiêu chí: NSE CV (trung bình 3 fold).
2. Thực nghiệm trên CV với siêu tham số đã chọn: E1–E4 (vai trò Hủa Na), thực nghiệm phụ (F4, F5, EMA), E5 (trọng số
   $w=(1+Q/Q_{95,train})^\gamma$, γ ∈ {1, 2}, chặn trên ∈ {không, 20}, chọn theo CV), E6 / E56 (phạt dự báo thiếu — chỉ LSTM, GRU),
   FUT (mưa thực đo t+1, cận trên).
3. Huấn luyện lại trên 2019–2024 (số epoch / số cây = trung vị vòng tốt nhất ở CV), dự báo tập test **khoá** 01/2025 – 08/2026.
4. Bài toán mở rộng t+1 … t+10 cho E1, E2, E5 (cùng siêu tham số)."""


NAME = {'persistence': 'Persistence', 'ridge': 'Ridge-ARX', 'extra_trees': 'Extra Trees', 'xgboost': 'XGBoost', 'rnn': 'RNN', 'lstm': 'LSTM', 'gru': 'GRU'}


def model_cells(models, title, intro):
    C = [md(f"# {title}\n\n{intro}\n\n{PROTOCOL}"),
         code(SETUP + "\nfrom src import pipeline\nfrom src.training.train import DISPLAY")]
    for m in models:
        C += [md(f"## {NAME[m]} — dò siêu tham số"), code(f"""tab, best = pipeline.tune(cfg, '{m}', verbose=False)
print(DISPLAY['{m}'], '→ siêu tham số chọn:', best)
display(tab.head(10).round(3))"""),
              md(f"### {NAME[m]}: thực nghiệm trên CV (t+1)"), code(f"""cvt = pipeline.cv_experiments(cfg, '{m}', verbose=False)
display(cvt.round(3)); print('Cấu hình:', pipeline.load_conf(cfg, '{m}'))"""),
              md(f"### {NAME[m]}: huấn luyện cuối 2019–2024 và test khoá (t+1)"), code(f"""met = pipeline.final_test(cfg, '{m}', verbose=False)
cols = ['NSE', 'KGE', 'r2', 'RMSE', 'MAE', 'PBIAS', 'NSE_Q95', 'RMSE_Q95', 'Peak_RMSE', 'Peak_Bias', 'Peak_Timing', 'n_iter']
display(met.groupby('exp', sort=False)[cols].agg(['mean', 'std']).round(3))"""),
              md(f"### {NAME[m]}: thí nghiệm A — cửa sổ chung L = 3"), code(f"""print(DISPLAY['{m}'], '→', pipeline.run_common_L(cfg, '{m}', 3, verbose=False))"""),
              md(f"### {NAME[m]}: bài toán mở rộng t+1 … t+10"), code(f"""cv10, met10 = pipeline.extended(cfg, '{m}', verbose=False)
display(cv10.round(3)); display(met10.groupby('exp', sort=False)[['NSE', 'KGE', 'RMSE', 'PBIAS', 'NSE_Q95', 'Peak_Bias']].mean().round(3))""")]
    return C


C = model_cells(['persistence', 'ridge', 'extra_trees', 'xgboost'], '03. Các mô hình baseline',
                r"""- **Persistence**: $\hat Q(t+h) = Q(t)$ — dự báo ngây thơ.
- **Ridge-ARX**: hồi quy tuyến tính L2 trên các độ trễ của Q, mưa, Hủa Na (X_flat); dò L và α.
- **Extra Trees**: rừng cây cực ngẫu nhiên (300 cây); dò L, min_samples_leaf, max_features.
- **XGBoost**: tăng cường gradient (hist, lr 0,05, dừng sớm 50 vòng); dò L, max_depth, min_child_weight.

Mô hình dạng bảng hồi quy trên cùng đích log(1+Q) chuẩn hoá (bình phương sai số), nhận trọng số mẫu ở E5. Cây quyết định không
ngoại suy vượt phạm vi đích đã học — cần lưu ý khi đánh giá các đỉnh lũ test vượt cực đại train.""")
save('03_Baseline.ipynb', C)
for i, m in [(4, 'rnn'), (5, 'lstm'), (6, 'gru')]:
    intro = MODEL_INTRO[m] + r"""

**Kiến trúc chung** (`src/models/recurrent.py`): chuỗi L × F → khối hồi quy (1–2 lớp, hidden ∈ {32, 64, 128}) → trạng thái ẩn
cuối → Dropout 0,2 → (ghép mưa tương lai ở FUT) → Linear(H). Huấn luyện: Huber có mặt nạ (× trọng số ở E5, + λ·phạt dự báo thiếu
ở E6), Adam 1e-3, batch 64, dừng sớm theo val (kiên nhẫn 20)."""
    save(f'0{i}_{m.upper()}.ipynb', model_cells([m], f'0{i}. {m.upper()}', intro))

# =====================================================================================================================
C = [md(r"""# 07. So sánh mô hình và trả lời ba câu hỏi nghiên cứu

1. LSTM và GRU có dự báo tốt hơn phương pháp truyền thống và học máy cơ bản không?
2. Dữ liệu vận hành Hủa Na có cải thiện dự báo không, biến nào có giá trị nhất?
3. Cách xử lý dữ liệu và huấn luyện ảnh hưởng thế nào, đặc biệt với lưu lượng lớn và mùa lũ?

Mọi so sánh dùng cùng ngày đích; bootstrap khối dịch chuyển (khối 30 ngày, 2000 lần) dùng **chung một bộ chỉ số** cho mọi cặp.
Mô hình / cấu hình được chọn theo **CV**; tập test chỉ để báo cáo."""),
     code(SETUP + r"""
from src.evaluation import comparison as C
from src.evaluation.metrics import LABEL, DIGITS
from src.training.train import DISPLAY, MODEL_NAMES
from src import pipeline
from src import results_store as S
OUT = results_dir(cfg, 'comparison')
cv, met, pred = C.load_all(cfg, 'H1')
cv10, met10, pred10 = C.load_all(cfg, 'H10')
M7 = MODEL_NAMES; DL = ['rnn', 'lstm', 'gru']; BL = ['persistence', 'ridge', 'extra_trees', 'xgboost']
conf = {m: pipeline.load_conf(cfg, m) for m in M7}
display(pd.DataFrame({DISPLAY[m]: {'siêu tham số': c.get('hp'), 'trọng số E5': c.get('weights'), 'λ': c.get('lambda')} for m, c in conf.items()}).T)"""),
     md(r"""## 1. Kết quả CV (t+1): NSE trung bình 3 fold"""),
     code(r"""cvm = C.cv_matrix(cv);
order = [c for c in ['PERS', 'E1', 'E2', 'E3', 'E4', 'P_F4', 'P_F5', 'P_EMA', 'W_g1_cnone', 'W_g2_cnone', 'W_g1_c20', 'W_g2_c20', 'E5', 'PEN_l0.5', 'PEN_l1.0', 'E6', 'E56', 'FUT'] if c in cvm.columns]
cvm = cvm[order]; display(cvm.round(3)); cvm.to_csv(OUT / 'cv_nse_H1.csv', encoding='utf-8-sig')
cvq = C.cv_matrix(cv, 'NSE Q>Q95 CV')[order]; print('NSE Q>Q95 (CV):'); display(cvq.round(3)); cvq.to_csv(OUT / 'cv_nse_q95_H1.csv', encoding='utf-8-sig')
pers = cvm.loc['Persistence', 'PERS']
FIN = ['E1', 'E2', 'E3', 'E4', 'P_F4', 'P_F5', 'P_EMA', 'E5', 'E6', 'E56']
best_cv = {m: cvm.loc[DISPLAY[m], [c for c in FIN if c in cvm.columns]].dropna().idxmax() for m in M7 if m != 'persistence'}
best_cv['persistence'] = 'PERS'
print('Cấu hình tốt nhất theo CV (không tính FUT):', {DISPLAY[m]: e for m, e in best_cv.items()})"""),
     md(r"""## 2. Câu hỏi 1 — LSTM/GRU so với baseline (test khoá, t+1)"""),
     code(r"""KEYM = ['NSE', 'KGE', 'r2', 'RMSE', 'MAE', 'PBIAS', 'NSE_Q95', 'RMSE_Q95', 'Peak_RMSE', 'Peak_Bias', 'Peak_Timing']
met_p = met.copy(); met_p.loc[met_p.model == 'persistence', 'exp'] = 'E2'      # Persistence không dùng biến → đặt cạnh E2
t1 = C.test_table(met_p, ['E2'], KEYM); display(t1); t1.to_csv(OUT / 'test_E2_H1.csv', encoding='utf-8-sig')
rows = []
for m in M7:
    e = best_cv[m]; d = met[(met.model == m) & (met.exp == e)]
    rows.append({'mô hình': DISPLAY[m], 'cấu hình tốt nhất (CV)': e, 'NSE CV': cvm.loc[DISPLAY[m], e], **{LABEL[k]: C.fmt(d[k], DIGITS[k]) for k in KEYM}})
tb = pd.DataFrame(rows).set_index('mô hình'); display(tb); tb.to_csv(OUT / 'test_best_cv_H1.csv', encoding='utf-8-sig')
pairs = []
for a in ['lstm', 'gru']:
    for b in BL:
        pairs.append((a, 'E2', b, 'PERS' if b == 'persistence' else 'E2', f'{DISPLAY[a]} − {DISPLAY[b]} (E2)'))
    pairs.append((a, 'E2', 'rnn', 'E2', f'{DISPLAY[a]} − RNN (E2)'))
pairs += [('lstm', 'E2', 'gru', 'E2', 'LSTM − GRU (E2)'), ('lstm', 'E5', 'gru', 'E5', 'LSTM − GRU (E5)')]
for a in ['lstm', 'gru']:
    for b in ['ridge', 'xgboost']:
        pairs.append((a, best_cv[a], b, best_cv[b], f'{DISPLAY[a]} {best_cv[a]} − {DISPLAY[b]} {best_cv[b]} (tốt nhất theo CV)'))
bq1 = C.bootstrap_table(cfg, pred, pairs); display(bq1); bq1.to_csv(OUT / 'bootstrap_cau1.csv', index=False, encoding='utf-8-sig')"""),
     md(r"""## 3. Câu hỏi 2 — vai trò dữ liệu Hủa Na"""),
     code(r"""hn = cvm[['E1', 'E2', 'E3', 'E4', 'P_F4', 'P_F5']].drop('Persistence')
print('NSE CV:'); display(hn.round(3))
print('Δ so với E1 (CV):'); display(hn.sub(hn['E1'], axis=0).drop(columns='E1').round(3))
pairs = [(m, e, m, 'E1', f'{DISPLAY[m]}: {e} − E1') for m in M7 if m != 'persistence' for e in ['E2', 'E3', 'E4']]
bq2 = C.bootstrap_table(cfg, pred, pairs); display(bq2); bq2.to_csv(OUT / 'bootstrap_cau2_huana.csv', index=False, encoding='utf-8-sig')
t2 = C.test_table(met, ['E1', 'E2', 'E3', 'E4'], ['NSE', 'KGE', 'NSE_Q95', 'Peak_Bias']); t2.to_csv(OUT / 'test_huana_H1.csv', encoding='utf-8-sig')
piv = met[met.exp.isin(['E1', 'E2', 'E3', 'E4', 'P_F4', 'P_F5'])].groupby(['model', 'exp']).NSE.mean().unstack()
piv = piv.reindex([m for m in M7 if m in piv.index]); piv.index = [DISPLAY[m] for m in piv.index]; print('NSE test:'); display(piv.round(3))
fig, ax = plt.subplots(figsize=(10, 3.8)); xs = np.arange(len(piv)); w = .2
for j, e in enumerate(['E1', 'E2', 'E3', 'E4']):
    ax.bar(xs + (j - 1.5) * w, cvm.loc[piv.index, e], w, label=f'{e} (CV)')
ax.set_xticks(xs); ax.set_xticklabels(piv.index); ax.set_ylabel('NSE CV'); ax.legend(fontsize=8, ncol=4); ax.set_ylim(0, None)
ax.set_title('Vai trò dữ liệu Hủa Na (CV, t+1)'); plt.tight_layout(); plt.savefig(OUT / 'fig_huana_cv.png'); plt.show()"""),
     md(r"""## 4. Câu hỏi 3 — xử lý dữ liệu và huấn luyện

### 4.1 Ngưỡng khoảng thiếu (từ notebook 02)"""),
     code(r"""gs = pd.read_csv(results_dir(cfg, 'dataset') / 'chon_nguong_khoang_thieu.csv'); display(gs.round(3))
print('G chọn =', dataset_meta(cfg)['max_gap'])"""),
     md(r"""### 4.2 Hàm mất mát có trọng số (E5), phạt dự báo thiếu (E6, E56), biến làm trơn (P_EMA), mưa tương lai (FUT)"""),
     code(r"""print('Δ NSE CV so với E2:'); display(cvm.drop('Persistence')[[c for c in ['E5', 'E6', 'E56', 'P_EMA', 'FUT'] if c in cvm]].sub(cvm.drop('Persistence')['E2'], axis=0).round(3))
print('Δ NSE Q>Q95 CV so với E2:'); display(cvq.drop('Persistence')[[c for c in ['E5', 'E6', 'E56', 'P_EMA', 'FUT'] if c in cvq]].sub(cvq.drop('Persistence')['E2'], axis=0).round(3))
pairs = [(m, 'E5', m, 'E2', f'{DISPLAY[m]}: E5 − E2') for m in M7 if m != 'persistence']
pairs += [(m, e, m, 'E2', f'{DISPLAY[m]}: {e} − E2') for m in ['lstm', 'gru'] for e in ['E6', 'E56']]
pairs += [(m, 'P_EMA', m, 'E2', f'{DISPLAY[m]}: P_EMA − E2') for m in M7 if m != 'persistence']
pairs += [(m, 'FUT', m, 'E2', f'{DISPLAY[m]}: FUT − E2 (cận trên)') for m in M7 if m != 'persistence']
bq3 = C.bootstrap_table(cfg, pred, pairs); display(bq3); bq3.to_csv(OUT / 'bootstrap_cau3.csv', index=False, encoding='utf-8-sig')
t3 = C.test_table(met, ['E2', 'E5', 'E6', 'E56'], ['NSE', 'KGE', 'PBIAS', 'UNDER', 'NSE_Q95', 'RMSE_Q95', 'UNDER_Q95', 'Peak_RMSE', 'Peak_Bias', 'Peak_Timing'])
display(t3); t3.to_csv(OUT / 'test_luu_luong_cao_H1.csv', encoding='utf-8-sig')"""),
     md(r"""### 4.3 Mùa lũ 2025 (t+1, trung bình các hạt giống)"""),
     code(r"""obs = pd.Series(dtype=float)
fig, axs = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
a, b = pd.Timestamp('2025-07-01'), pd.Timestamp('2025-10-31')
for ax, e in zip(axs, ['E2', 'E5']):
    first = True
    for m, c in [('persistence', '0.6'), ('ridge', 'C2'), ('xgboost', 'C1'), ('lstm', 'C3'), ('gru', 'C0')]:
        ee = 'PERS' if m == 'persistence' else e
        r = C.arrays(pred, m, ee)
        if r is None: continue
        O, M_, P, idx = r; s = pd.Series(np.mean(P, axis=0)[:, 0], index=idx + pd.Timedelta(days=1))
        if first: ax.plot(pd.Series(O[:, 0], index=idx + pd.Timedelta(days=1)).loc[a:b], 'k', lw=1.3, label='quan trắc'); first = False
        ax.plot(s.loc[a:b], color=c, lw=1, label=DISPLAY[m])
    ax.set_title(f'{e}: {pipeline.EXP_NAME[e]}'); ax.set_ylabel('Q đến CĐ (m³/s)'); ax.legend(fontsize=8, ncol=6)
plt.tight_layout(); plt.savefig(OUT / 'fig_mua_lu_2025_H1.png'); plt.show()"""),
     md(r"""## 5. Bài toán mở rộng t+1 … t+10"""),
     code(r"""met10_p = met10.copy(); met10_p.loc[met10_p.model == 'persistence', 'exp'] = 'E2'
t10 = C.test_table(met10_p, ['E1', 'E2', 'E5'], ['NSE', 'KGE', 'RMSE', 'PBIAS', 'NSE_Q95', 'Peak_Bias'])
display(t10); t10.to_csv(OUT / 'test_H10.csv', encoding='utf-8-sig')
from src.evaluation.metrics import per_horizon
rows = []
for m in M7:
    for e in (['PERS'] if m == 'persistence' else ['E2', 'E5']):
        r = C.arrays(pred10, m, e)
        if r is None: continue
        O, M_, P, _ = r
        ph = pd.concat([per_horizon(O, p, M_)[['MAE', 'NSE']] for p in P]).groupby(level=0).mean()
        for h, x in ph.iterrows(): rows.append({'mô hình': DISPLAY[m], 'thực nghiệm': e, 'h': h, **x.to_dict()})
ph = pd.DataFrame(rows); ph.to_csv(OUT / 'theo_h_H10.csv', index=False, encoding='utf-8-sig')
fig, axs = plt.subplots(1, 2, figsize=(13, 4))
for (m, e), d in ph.groupby(['mô hình', 'thực nghiệm'], sort=False):
    ls = '-' if e != 'E5' else '--'
    for ax, k in zip(axs, ['MAE', 'NSE']): ax.plot(d.h, d[k], ls, marker='o', ms=3, label=f'{m} {e}')
axs[0].set_ylabel('MAE test (m³/s)'); axs[1].set_ylabel('NSE test'); axs[1].legend(fontsize=7, ncol=2)
for ax in axs: ax.set_xlabel('h (ngày)'); ax.set_xticks(range(1, 11))
plt.tight_layout(); plt.savefig(OUT / 'fig_theo_h_H10.png'); plt.show()
pairs = [(a, 'E2', b, 'PERS' if b == 'persistence' else 'E2', f'{DISPLAY[a]} − {DISPLAY[b]} (E2, 10 ngày)') for a in ['lstm', 'gru'] for b in BL + ['rnn']]
pairs += [('lstm', 'E2', 'gru', 'E2', 'LSTM − GRU (E2, 10 ngày)')] + [(m, 'E2', m, 'E1', f'{DISPLAY[m]}: E2 − E1 (10 ngày)') for m in M7 if m != 'persistence']
pairs += [(m, 'E5', m, 'E2', f'{DISPLAY[m]}: E5 − E2 (10 ngày)') for m in M7 if m != 'persistence']
b10 = C.bootstrap_table(cfg, pred10, pairs); display(b10); b10.to_csv(OUT / 'bootstrap_H10.csv', index=False, encoding='utf-8-sig')"""),
     md(r"""## 6. Thí nghiệm A (L = 3 chung) và thí nghiệm B (L chọn theo CV)

**A — so sánh có kiểm soát:** mọi mô hình dùng L = 3 (siêu tham số khác dò lại với L cố định). **B — cấu hình tối ưu:** L ∈ {3, 7,
14, 30} chọn theo CV cho từng mô hình. Mô hình nào ở B đã chọn L = 3 thì A ≡ B (quy trình tất định, không chạy lại)."""),
     code(r"""cvA, metA, predA = C.load_all(cfg, 'H1', variant='L3')
L_of = {m: (pipeline.load_conf(cfg, m, 'L3') if not pipeline.load_conf(cfg, m, 'L3').get('same_as_base') else conf[m])['hp'].get('L') for m in M7}
print('L dùng ở thí nghiệm A:', {DISPLAY[m]: L_of[m] for m in M7}, '| ở thí nghiệm B:', {DISPLAY[m]: conf[m]['hp'].get('L') for m in M7})
mA = metA.copy(); mA.loc[mA.model == 'persistence', 'exp'] = 'E2'
tA = C.test_table(mA, ['E1', 'E2', 'E3', 'E4'], ['NSE', 'KGE', 'RMSE', 'MAE', 'PBIAS'])
print('Thí nghiệm A — test (t+1):'); display(tA); tA.to_csv(OUT / 'thi_nghiem_A_L3_test.csv', encoding='utf-8-sig')
cvmA = C.cv_matrix(cvA)[[c for c in ['PERS', 'E1', 'E2', 'E3', 'E4', 'E5'] if c in C.cv_matrix(cvA).columns]]
print('Thí nghiệm A — NSE CV:'); display(cvmA.round(3)); cvmA.to_csv(OUT / 'thi_nghiem_A_L3_cv.csv', encoding='utf-8-sig')
pairs = [(a, 'E2', b, 'PERS' if b == 'persistence' else 'E2', f'[A, L=3] {DISPLAY[a]} − {DISPLAY[b]} (E2)') for a in ['lstm', 'gru'] for b in BL + ['rnn']]
bA = C.bootstrap_multi(cfg, predA, pairs); display(bA); bA.to_csv(OUT / 'thi_nghiem_A_bootstrap.csv', encoding='utf-8-sig')"""),
     md(r"""## 7. Kiểm định chéo theo từng fold và quy tắc một sai số chuẩn

Chỉ tiêu từng fold (trung bình hạt giống), trung bình và độ lệch chuẩn giữa 3 fold, thời gian huấn luyện một lần, số vòng.
**Quy tắc 1-SE** (khai báo trước, chỉ là phân tích phụ — lựa chọn chính vẫn theo NSE CV cao nhất): trong các cấu hình có
NSE CV ≥ NSE tốt nhất − SE (SE = ĐLC giữa 3 fold / √3), chọn cấu hình đơn giản nhất (L nhỏ, ít tham số / α lớn)."""),
     code(r"""ft = C.fold_table(cv, ['PERS', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E56'])
display(ft.round(3)); ft.to_csv(OUT / 'cv_theo_fold_H1.csv', encoding='utf-8-sig')
rows = []
SIMPLE = {'ridge': (['L', 'alpha'], [True, False]), 'extra_trees': (['L', 'min_samples_leaf'], [True, False]),
          'xgboost': (['L', 'max_depth'], [True, True]), 'rnn': (['L', 'layers', 'hidden'], [True, True, True]),
          'lstm': (['L', 'layers', 'hidden'], [True, True, True]), 'gru': (['L', 'layers', 'hidden'], [True, True, True])}
for m, (keys, asc) in SIMPLE.items():
    tab = S.read_tuning(cfg, m)
    best = tab.iloc[0]; se = best['NSE CV (ĐLC giữa fold)'] / np.sqrt(3)
    pick = tab[tab['NSE CV'] >= best['NSE CV'] - se].sort_values(keys, ascending=asc).iloc[0]
    rows.append({'mô hình': DISPLAY[m], 'chọn (NSE CV cao nhất)': {k: best[k] for k in keys}, 'NSE CV': best['NSE CV'], 'SE': se,
                 'chọn theo 1-SE': {k: pick[k] for k in keys}, 'NSE CV (1-SE)': pick['NSE CV'], 'trùng?': all(pick[k] == best[k] for k in keys)})
ose = pd.DataFrame(rows).set_index('mô hình'); display(ose); ose.to_csv(OUT / 'quy_tac_1se.csv', encoding='utf-8-sig')"""),
     md(r"""## 8. Bootstrap nhiều chỉ tiêu (ΔNSE, ΔKGE, ΔMAE, ΔRMSE, tỉ lệ A tốt hơn B)"""),
     code(r"""pairs = [(a, 'E2', b, 'PERS' if b == 'persistence' else 'E2', f'{DISPLAY[a]} − {DISPLAY[b]} (E2)') for a in ['lstm', 'gru'] for b in BL + ['rnn']]
pairs.append(('lstm', 'E2', 'gru', 'E2', 'LSTM − GRU (E2)'))
b1 = C.bootstrap_multi(cfg, pred, pairs); display(b1); b1.to_csv(OUT / 'bootstrap_nhieu_chi_tieu_cau1.csv', encoding='utf-8-sig')
pairs = [(m, e, m, 'E1', f'{DISPLAY[m]}: {e} − E1') for m in M7 if m != 'persistence' for e in ['E2', 'E3', 'E4']]
b2 = C.bootstrap_multi(cfg, pred, pairs); display(b2); b2.to_csv(OUT / 'bootstrap_nhieu_chi_tieu_huana.csv', encoding='utf-8-sig')
pairs = [(m, 'E5', m, 'E2', f'{DISPLAY[m]}: E5 − E2') for m in M7 if m != 'persistence'] + \
        [(m, e, m, 'E2', f'{DISPLAY[m]}: {e} − E2') for m in ['lstm', 'gru'] for e in ['E6', 'E56']]
b3 = C.bootstrap_multi(cfg, pred, pairs); display(b3); b3.to_csv(OUT / 'bootstrap_nhieu_chi_tieu_ham_mat_mat.csv', encoding='utf-8-sig')
print('Δ theo fold (CV) của E2/E3/E4 so với E1 — phụ thuộc giai đoạn thủy văn:')
g = cv[cv.exp.isin(['E1', 'E2', 'E3', 'E4'])].groupby(['model', 'exp', 'fold']).NSE.mean().unstack('exp')
dfold = g[['E2', 'E3', 'E4']].sub(g['E1'], axis=0).round(3); dfold.index = dfold.index.set_levels([DISPLAY[m] for m in dfold.index.levels[0]], level=0)
display(dfold.unstack('fold')); dfold.to_csv(OUT / 'huana_delta_theo_fold.csv', encoding='utf-8-sig')"""),
     md(r"""## 9. Độ ổn định giữa hạt giống, đỉnh lũ từng sự kiện, độ nhạy với 102 ngày răng cưa chưa rõ nguyên nhân

Ba đỉnh test là **mô tả**, không phải bằng chứng thống kê. Độ nhạy: bỏ các ngày đích bị cờ "nghi nhiễu tính toán" (tên cờ cũ; Q đến CĐ
đã được xác nhận là số liệu đo) **khi chấm điểm** (nhãn và huấn luyện giữ nguyên) — trên test và trên val của CV."""),
     code(r"""st = met[met.exp.isin(['E2', 'E5', 'E6', 'E56'])].groupby(['model', 'exp']).agg(NSE_TB=('NSE', 'mean'), NSE_ĐLC=('NSE', 'std'),
        NSE_min=('NSE', 'min'), NSE_max=('NSE', 'max'), số_hạt_giống=('NSE', 'size')).round(3)
st.index = st.index.set_levels([DISPLAY[m] for m in st.index.levels[0]], level=0); display(st); st.to_csv(OUT / 'on_dinh_hat_giong.csv', encoding='utf-8-sig')
peaks = [pd.Timestamp(p) for p in dataset_meta(cfg)['test_peaks']]
pk = C.peak_table(pred, peaks, [('persistence', 'PERS')] + [(m, e) for m in M7 if m != 'persistence' for e in ['E2', 'E5']])
display(pk.round(1)); pk.to_csv(OUT / 'dinh_lu_tung_su_kien.csv', index=False, encoding='utf-8-sig')
from src.data.freeze import frozen_dir
flags = pd.read_csv(frozen_dir(cfg) / 'qc_flags.csv', parse_dates=['date']).date
me = [('persistence', 'PERS')] + [(m, 'E2') for m in M7 if m != 'persistence']
sens = C.sensitivity_excluding(pred, flags, me); print('Test:'); display(sens.round(3))
cvp = S.read_pred(cfg, 'cv', 1)                    # dự báo trên tập xác thực của 3 fold, mọi mô hình (results/du_bao.csv.gz)
rows = []
for m, e in me:
    for f in ['F1', 'F2', 'F3']:
        d = cvp[(cvp.model == m) & (cvp.exp == e) & (cvp.fold == f)]
        if not len(d): continue
        r = C.sensitivity_excluding(d.assign(model=m, exp=e), flags, [(m, e)]).iloc[0].to_dict(); r['fold'] = f; rows.append(r)
sens_cv = pd.DataFrame(rows); print('CV (val):'); display(sens_cv.round(3))
pd.concat({'test': sens, 'cv': sens_cv}).to_csv(OUT / 'do_nhay_102_ngay_nghi_nhieu.csv', encoding='utf-8-sig')"""),
     md(r"""## 10. Chi phí và số tham số"""),
     code(r"""rows = []
for m in M7:
    c = conf[m]; rows.append({'mô hình': DISPLAY[m], 'siêu tham số': c.get('hp'), 'vòng (E2, test)': met[(met.model == m) & (met.exp.isin(['E2', 'PERS']))].n_iter.median()})
display(pd.DataFrame(rows).set_index('mô hình'))"""),
     md(r"""## 11. Hình tổng hợp so sánh mô hình trên tập test (t+1, E2)

(a) NSE test của từng mô hình: mỗi điểm nhỏ là một hạt giống, vạch đứng là trung bình. (b) ΔNSE của LSTM và GRU so với từng
mô hình khác, kèm khoảng tin cậy 95 % bootstrap khối (`bootstrap_cau1.csv`); khoảng cắt qua 0 nghĩa là chưa có ý nghĩa thống kê."""),
     code(r"""DLC, BLC, INK = '#2a78d6', '#8a8a85', '#3d3d3a'
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={'width_ratios': [1, 1.25]})
ax = axs[0]; names = [DISPLAY[m] for m in M7]
for i, m in enumerate(M7):
    d = met[(met.model == m) & (met.exp == ('PERS' if m == 'persistence' else 'E2'))].NSE.values
    col = DLC if m in ('lstm', 'gru') else BLC
    ax.scatter(d, np.full(len(d), i), s=22, color=col, alpha=.6, zorder=3)
    ax.plot([d.mean()] * 2, [i - .3, i + .3], color=col, lw=2.5, zorder=4)
    ax.text(d.mean(), i - .38, f'{d.mean():.3f}', fontsize=8, color=INK, ha='center')
ax.set_yticks(range(len(M7))); ax.set_yticklabels(names); ax.invert_yaxis(); ax.set_xlabel('NSE test (t+1, E2)')
ax.set_title('(a) NSE test theo mô hình', loc='left', fontsize=10)
ax = axs[1]
b = pd.read_csv(OUT / 'bootstrap_cau1.csv', encoding='utf-8-sig'); b = b[b['so sánh'].str.endswith('(E2)') & ~b['so sánh'].str.startswith('LSTM − GRU')]
lo = b['KTC 95 %'].str.extract(r'\[([-+0-9.]+); ([-+0-9.]+)\]').astype(float)
for i, (_, r) in enumerate(b.iterrows()):
    col = DLC if r['so sánh'].startswith('LSTM') else '#eb6834'
    ax.plot([lo.loc[r.name, 0], lo.loc[r.name, 1]], [i, i], color=col, lw=2)
    ax.scatter(r['ΔNSE (A − B)'], i, s=40, color=col, zorder=3)
ax.axvline(0, color=INK, lw=.8); ax.set_yticks(range(len(b))); ax.set_yticklabels(b['so sánh'].str.replace(' (E2)', '', regex=False), fontsize=8)
ax.invert_yaxis(); ax.set_xlabel('ΔNSE test (A − B), KTC 95 % bootstrap'); ax.set_title('(b) LSTM / GRU so với các mô hình khác', loc='left', fontsize=10)
plt.tight_layout(); plt.savefig(OUT / 'fig_so_sanh_test_H1.png', dpi=150); plt.show()"""),
     md(r"""## 12. Vai trò dữ liệu Hủa Na theo mùa và trạng thái dòng chảy (t+1, dự báo đã lưu)

MAE của E1 … E4 tách theo **ngày đích**: mùa lũ (T7–T10) / mùa kiệt; dòng chảy quan trắc thấp (< Q50), trung bình (Q50–Q90),
cao (≥ Q90). Ngưỡng Q50 = 58 và Q90 = 173 m³/s lấy từ 2019–2023 (như notebook 01, cố định, không dùng dữ liệu test). Dự báo là
trung bình các hạt giống. Hai nguồn: **val của CV** (3 fold gộp — tập dùng để lựa chọn) và **test** (chỉ báo cáo). Chỉ dùng MAE
vì NSE trong nhóm con phụ thuộc mạnh vào phương sai của nhóm. Đây là phân tích mô tả, không kiểm định."""),
     code(r"""daily_ = pd.read_csv(ROOT / cfg['paths']['processed'] / 'cuadat_huana_mua_daily.csv', parse_dates=['date'], index_col='date')
q_ = daily_.q_cd[(daily_.index <= '2023-12-31') & ~daily_.cd_missing]; Q50, Q90 = q_.quantile(.5), q_.quantile(.9)
print(f'Q50 = {Q50:.1f}, Q90 = {Q90:.1f} m³/s (2019–2023)')
def groups(dt, o):
    fl = dt.month.isin([7, 8, 9, 10])
    return {'cả năm': np.ones(len(o), bool), 'mùa lũ T7–T10': fl, 'mùa kiệt T11–T6': ~fl,
            f'Q thấp (< {Q50:.0f})': o < Q50, f'Q trung bình ({Q50:.0f}–{Q90:.0f})': (o >= Q50) & (o < Q90), f'Q cao (≥ {Q90:.0f})': o >= Q90}
rows = []
for src_, data in [('val CV (3 fold)', cvp[cvp.h == 1]), ('test', pred[pred.h == 1])]:
    for m in M7:
        if m == 'persistence': continue
        for e in ['E1', 'E2', 'E3', 'E4']:
            d = data[(data.model == m) & (data.exp == e) & data.dich_that.astype(bool)]
            keys = ['fold', 'ngay_goc'] if 'fold' in d else ['ngay_goc']
            a = d.groupby(keys).agg(o=('Q_do', 'first'), p=('Q_du_bao', 'mean')).reset_index()
            dt = pd.DatetimeIndex(a.ngay_goc) + pd.Timedelta(days=1)
            for g, mk in groups(dt, a.o.values).items():
                rows.append({'nguồn': src_, 'mô hình': DISPLAY[m], 'thực nghiệm': e, 'nhóm': g, 'n': int(mk.sum()),
                             'MAE': float(np.abs(a.p.values[mk] - a.o.values[mk]).mean())})
hs = pd.DataFrame(rows)
piv = hs.pivot_table(index=['nguồn', 'nhóm', 'mô hình'], columns='thực nghiệm', values='MAE')
for e in ['E2', 'E3', 'E4']: piv[f'Δ% {e}'] = 100 * (piv[e] / piv['E1'] - 1)
piv['n'] = hs[hs['thực nghiệm'] == 'E1'].set_index(['nguồn', 'nhóm', 'mô hình']).n
display(piv.round(1)); piv.to_csv(OUT / 'huana_theo_mua_trang_thai.csv', encoding='utf-8-sig')
print('Δ% MAE so với E1, trung vị 6 mô hình:')
med = piv.groupby(level=['nguồn', 'nhóm'], sort=False)[['Δ% E2', 'Δ% E3', 'Δ% E4']].median()
display(med.round(1)); med.to_csv(OUT / 'huana_theo_mua_trang_thai_trung_vi.csv', encoding='utf-8-sig')""")]
save('07_SoSanh.ipynb', C)


# =====================================================================================================================
C = [md(r"""# 08. Điều tra lần chạy LSTM-E5 bất thường

Ở huấn luyện cuối (2019–2024, số epoch cố định = trung vị epoch tốt nhất trên CV), 1/5 hạt giống của LSTM-E5 cho NSE test
thấp bất thường. Notebook này kiểm tra: NaN/Inf trong dữ liệu và gradient, nổ gradient, chuẩn trọng số, hàm mất mát theo
epoch, trọng số mẫu, độ khớp trên chính tập train, khả năng tái lập, và **tần suất hỏng của quy trình số epoch cố định trên val
của CV** (không dùng test). Không loại hạt giống khỏi kết quả."""),
     code(SETUP + r"""
import torch
from src import pipeline
from src.data.feature_engineering import feature_list
from src.training.train import fit_predict
from src.training.losses import flow_weights
from src.evaluation.metrics import summary
OUT = results_dir(cfg, 'models', 'lstm', 'dieu_tra_E5')
from src import results_store as S
conf = pipeline.load_conf(cfg, 'lstm'); print(conf)
log = pd.read_csv(pipeline.out_dir(cfg, 'lstm') / 'train_log_H1.csv')
met = S.read_metrics(cfg, 'test', 1, 'chinh', 'lstm')
display(met[met.exp.isin(['E2', 'E5', 'E6'])][['exp', 'seed', 'n_iter', 'NSE', 'PBIAS', 'NSE_Q95', 'Peak_Bias']].round(3))"""),
     md(r"""## 1. NaN / Inf, gradient, chuẩn trọng số, hàm mất mát theo epoch (huấn luyện cuối)"""),
     code(r"""print('Số batch có NaN/Inf (loss hoặc gradient) theo thực nghiệm:', log.groupby('exp').nonfinite_batches.sum().to_dict())
last = log.sort_values('epoch').groupby(['exp', 'seed']).tail(1).set_index(['exp', 'seed'])
display(last.loc[['E2', 'E5', 'E6'], ['epoch', 'train_loss', 'train_loss_max_batch', 'grad_norm_mean', 'grad_norm_max', 'weight_norm']].round(3))
fig, axs = plt.subplots(1, 3, figsize=(15, 3.6))
for (e, s), d in log[log.exp == 'E5'].groupby(['exp', 'seed']):
    for ax, k in zip(axs, ['train_loss', 'grad_norm_max', 'weight_norm']):
        ax.plot(d.epoch, d[k], marker='o', ms=3, label=f'hạt giống {s}', lw=2.5 if s == met.loc[met.exp.eq('E5'), ['seed', 'NSE']].sort_values('NSE').seed.iloc[0] else 1)
for ax, k in zip(axs, ['hàm mất mát train (TB batch)', 'chuẩn gradient lớn nhất (trước khi cắt)', 'chuẩn trọng số']): ax.set_title(k); ax.set_xlabel('epoch')
axs[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(OUT / 'fig_log_huan_luyen_E5.png'); plt.show()"""),
     md(r"""## 2. Dữ liệu và trọng số mẫu của cấu hình E5 (giai đoạn cuối)"""),
     code(r"""ds = pipeline.get_dataset(cfg, 1); feats = feature_list(cfg, 'F1'); hp = conf['hp']
v = ds.view('FINAL', feats, hp['L'])
print('X, Y hữu hạn:', np.isfinite(v.X_seq).all(), np.isfinite(v.Y).all(), '| số mẫu train:', len(v.idx_train))
w = flow_weights(v.W_q, v.M, v.idx_train, v.q95, conf['weights']['gamma'], conf['weights']['cap'])
wt = w[v.idx_train][v.M[v.idx_train]]
print(f"γ = {conf['weights']['gamma']}, Q95 train = {v.q95:.1f}: trọng số train min {wt.min():.2f}, trung vị {np.median(wt):.2f}, max {wt.max():.2f}; "
      f"{(wt > 5).sum()} ô có w > 5 (chiếm {100 * wt[wt > 5].sum() / wt.sum():.1f} % tổng trọng số)")"""),
     md(r"""## 3. Tái lập và độ khớp trên chính tập train

Huấn luyện lại đúng cấu hình E5 cho từng hạt giống: (a) dự báo test phải trùng file đã lưu (tái lập); (b) NSE **trên tập train**
cho biết lần chạy hỏng là do tối ưu thất bại (khớp train kém) hay do quá khớp (khớp train tốt nhưng test kém)."""),
     code(r"""cvd = S.read_metrics(cfg, 'cv', 1, 'chinh', 'lstm'); n_e5 = max(1, int(round(cvd[cvd.exp == 'E5'].best_iter.median())))
saved = S.read_pred(cfg, 'test', 1, 'chinh', 'lstm')
rows = []
for s in cfg['seeds']['final']:
    P_te, _ = fit_predict(cfg, v, 'lstm', hp, s, conf['weights']['gamma'], conf['weights']['cap'], 0.0, n_iter=n_e5)
    P_tr, _ = fit_predict(cfg, v, 'lstm', hp, s, conf['weights']['gamma'], conf['weights']['cap'], 0.0, n_iter=n_e5, idx_eval=v.idx_train)
    old = saved[(saved.exp == 'E5') & (saved.seed == s)].sort_values('ngay_goc').Q_du_bao.values
    rows.append({'hạt giống': s, 'epoch': n_e5, 'tái lập (chênh lớn nhất, m³/s)': float(np.abs(P_te[:, 0] - old).max()),
                 'NSE train (trong mẫu)': summary(v.QO[v.idx_train], P_tr, v.M[v.idx_train])['NSE'],
                 'PBIAS train': summary(v.QO[v.idx_train], P_tr, v.M[v.idx_train])['PBIAS'],
                 'NSE test': summary(v.QO[v.idx_eval], P_te, v.M[v.idx_eval])['NSE']})
rep = pd.DataFrame(rows).set_index('hạt giống'); display(rep.round(3)); rep.to_csv(OUT / 'tai_lap_va_khop_train.csv', encoding='utf-8-sig')"""),
     md(r"""## 4. Độ ổn định của quy trình số epoch cố định — trên val của CV (không dùng test)

Mỗi fold: huấn luyện với **đúng số epoch cố định** như giai đoạn cuối (không dừng sớm), 10 hạt giống, cho E2, E5, E6; chấm trên
val của fold. Nếu E5 có tỉ lệ "hỏng" cao hơn → bất ổn nằm ở quy trình (trọng số lớn + không dừng sớm), không phải lỗi dữ liệu."""),
     code(r"""n_of = {e: max(1, int(round(cvd[cvd.exp == e].best_iter.median()))) for e in ['E2', 'E5', 'E6']}
spec = {'E2': (0, None, 0.0), 'E5': (conf['weights']['gamma'], conf['weights']['cap'], 0.0), 'E6': (0, None, conf['lambda'])}
rows = []
for f in ['F1', 'F2', 'F3']:
    vf = ds.view(f, feats, hp['L'])
    for e, (g_, c_, l_) in spec.items():
        for s in range(10):
            P, _ = fit_predict(cfg, vf, 'lstm', hp, s, g_, c_, l_, n_iter=n_of[e])
            rows.append({'fold': f, 'exp': e, 'seed': s, 'epoch': n_of[e], 'NSE val': summary(vf.QO[vf.idx_eval], P, vf.M[vf.idx_eval])['NSE']})
stab = pd.DataFrame(rows); stab.to_csv(OUT / 'on_dinh_epoch_co_dinh_cv.csv', index=False, encoding='utf-8-sig')
agg = stab.groupby(['fold', 'exp'])['NSE val'].agg(['mean', 'std', 'min', 'max'])
ref = stab.groupby(['fold', 'exp'])['NSE val'].transform('median')
stab['hỏng (< trung vị − 0,15)'] = stab['NSE val'] < ref - 0.15
agg['số lần hỏng /10'] = stab.groupby(['fold', 'exp'])['hỏng (< trung vị − 0,15)'].sum()
display(agg.round(3)); agg.to_csv(OUT / 'on_dinh_tom_tat.csv', encoding='utf-8-sig')
fig, ax = plt.subplots(figsize=(8, 3.6))
for i, (k, d) in enumerate(stab.groupby(['fold', 'exp'])):
    ax.scatter(np.full(len(d), i), d['NSE val'], s=12)
ax.set_xticks(range(len(agg))); ax.set_xticklabels([f'{a}\n{b}' for a, b in agg.index], fontsize=8); ax.set_ylabel('NSE val')
ax.set_title('LSTM, số epoch cố định, 10 hạt giống'); plt.tight_layout(); plt.savefig(OUT / 'fig_on_dinh_cv.png'); plt.show()""")]
C += [md(r"""## 5. Lần chạy hỏng sai ở đâu trên test? (chẩn đoán, không dùng để chọn hay sửa)

Phân rã tổng bình phương sai số theo ngày; đánh dấu các ngày mà đầu vào Q CĐ(t) **vượt cực đại train** (ngoài miền huấn luyện)."""),
      code(r"""p = saved[saved.exp == 'E5']
W = p.pivot(index='ngay_goc', columns='seed', values='Q_du_bao'); o = p[p.seed == p.seed.min()].set_index('ngay_goc').Q_do
E2_ = W.sub(o, axis=0) ** 2
qmax_tr = float(np.nanmax(v.QO[v.idx_train])); qin = pd.Series(v.qlast[v.idx_eval], index=v.dates[v.idx_eval])
print(f'Cực đại Q đích train = {qmax_tr:.0f} m³/s; ngày gốc test có Q(t) > cực đại train:', list(qin[qin > qmax_tr].index.date))
print('Tổng bình phương sai số theo hạt giống:', E2_.sum().round(-4).to_dict())
bad = E2_.sum().idxmax()
top = E2_[bad].nlargest(6)
tab = pd.concat([o, qin.rename('Q(t) đầu vào'), W], axis=1).loc[top.index]
tab['tỉ trọng SSE (hạt giống hỏng)'] = (top / E2_[bad].sum()).values
display(tab.round(2)); tab.to_csv(OUT / 'sai_so_lon_nhat_hat_giong_hong.csv', encoding='utf-8-sig')
nxt = qin.index[qin > qmax_tr]
print(f'Tỉ trọng SSE của hạt giống {bad} ở 7 ngày sau các ngày đầu vào ngoài miền:',
      round(float(E2_[bad].loc[[d for d in E2_.index if any(0 <= (d - x).days <= 6 for x in nxt)]].sum() / E2_[bad].sum()), 3))""")]
save('08_DieuTra_LSTM_E5.ipynb', C)


# =====================================================================================================================
# 09 — kịch bản bổ sung KH: giá trị của thông tin kế hoạch xả Hủa Na (thiết kế viết trước khi chạy)
C = [md(r"""# 09. Kịch bản bổ sung KH: giá trị của thông tin kế hoạch xả Hủa Na ngày t+1

Thiết kế, giả thuyết và cách diễn giải được viết **trước khi chạy** ở `docs/ThietKe_KichBan_KeHoachXa.md`.

Mọi biến thể = E2 (F1) + Q ra Hủa Na ngày t+1, ghép như FUT ghép mưa tương lai:
- **KH_TD:** huấn luyện và dự báo bằng Q ra t+1 thực đo (cận trên);
- **KH_AD{σ}:** mô hình KH_TD, khi dự báo nhận kế hoạch có sai số σ (tình huống hiện nay: chưa có lưu trữ kế hoạch xả);
- **KH_SS{σ}:** huấn luyện và dự báo bằng kế hoạch có sai số σ.

Sai số kế hoạch: Q̃ = max(0, Q·(1 + σ·z)), z ~ N(0, 1) theo ngày; σ ∈ {10, 20, 30, 50 %}.

Quy trình:
- siêu tham số giữ như E2, hàm mất mát cơ bản;
- 6 mô hình (không có Persistence);
- CV 3 fold × 3 hạt giống, huấn luyện cuối 5 hạt giống;
- **không có bước lựa chọn** giữa các biến thể, tập kiểm tra chấm một lần.

Mã: `src/plan_scenario.py` (không sửa `src/data/`, dữ liệu đóng băng v1 vẫn được xác minh)."""),
     code(SETUP + r"""
from src import pipeline, plan_scenario as K
from src import results_store as S
from src.evaluation import comparison as C
from src.evaluation.metrics import LABEL, DIGITS
from src.training.train import DISPLAY
from src.training.tuning import cv_summary
OUT = results_dir(cfg, 'ke_hoach_xa')
ds = pipeline.get_dataset(cfg, 1)
peaks, thr = pipeline.test_peaks(cfg, ds)
print('Đỉnh lũ test:', [d.date() for d in peaks], f'(ngưỡng Q99 train = {thr:.0f} m³/s)')
fi = ds.daily['hn_missing'].values; pos = ds.S['pos'].values
v_ = ds.view('FINAL', pipeline.feature_list(cfg, 'F1'), 3)
for nm, ix in [('train cuối', v_.idx_train), ('test', v_.idx_eval)]:
    print(f'{nm}: số mẫu có Q ra Hủa Na ngày t+1 thiếu (dùng giá trị gần nhất đã biết): {int(fi[pos[ix] + 1].sum())}')"""),
     md(r"""## 1. Kiểm tra trước khi chạy: đường mã mới tái lập E2 đã lưu

Huấn luyện lại E2 (không ghép biến) qua đúng các hàm dùng cho kịch bản KH, hạt giống 0, so với dự báo test E2 đã lưu."""),
     code(r"""rep = pd.DataFrame([{'mô hình': DISPLAY[m], 'chênh lớn nhất (m³/s)': K.check_e2_reproduction(cfg, ds, m)} for m in K.MODELS]).set_index('mô hình')
display(rep); rep.to_csv(OUT / 'tai_lap_E2.csv', encoding='utf-8-sig')
assert rep['chênh lớn nhất (m³/s)'].max() < 0.5"""),
     md(r"""## 2. Chạy CV và huấn luyện cuối"""),
     code(r"""cvs, cvp, mets, preds = [], [], [], []
for m in K.MODELS:
    cv_m, cp_m = K.run_cv(cfg, ds, m)
    met_m, pr_m = K.run_final(cfg, ds, m, cv_m, peaks)
    cvs.append(cv_m); cvp.append(cp_m); mets.append(met_m); preds.append(pr_m)
    # ghi vào các file gộp, nhóm 'KH': results/ket_qua_thuc_nghiem.csv (chỉ tiêu), results/du_bao.csv.gz (dự báo)
    for gd, a, b in [('cv', cv_m, cp_m), ('test', met_m, pr_m)]:
        S.write_metrics(cfg, a, m, 1, gd, 'KH'); S.write_pred(cfg, b, m, 1, gd, 'KH')
cvk, cvpk, metk, predk = pd.concat(cvs), pd.concat(cvp), pd.concat(mets), pd.concat(preds)"""),
     md(r"""## 3. Kết quả CV (NSE trung bình 3 fold), cùng E2 và FUT đã có"""),
     code(r"""cv0, met0, pred0 = C.load_all(cfg, 'H1')
EXPS = ['E2', 'FUT', 'KH_TD'] + [K.tag('AD', s) for s in K.SIGMAS] + [K.tag('SS', s) for s in K.SIGMAS]
cv_all = pd.concat([cv0[cv0.exp.isin(['E2', 'FUT']) & cv0.model.isin(K.MODELS)], cvk])
cvt = C.cv_matrix(cv_all)[EXPS]; display(cvt.round(3)); cvt.to_csv(OUT / 'cv_nse.csv', encoding='utf-8-sig')
cvq = C.cv_matrix(cv_all, 'NSE Q>Q95 CV')[EXPS]; print('NSE Q>Q95 (CV):'); display(cvq.round(3)); cvq.to_csv(OUT / 'cv_nse_q95.csv', encoding='utf-8-sig')"""),
     md(r"""## 4. Kết quả trên tập kiểm tra (TB ± ĐLC giữa các hạt giống)"""),
     code(r"""met_all = pd.concat([met0[met0.exp.isin(['E2', 'FUT']) & met0.model.isin(K.MODELS)], metk])
KEYS = ['NSE', 'KGE', 'RMSE', 'MAE', 'PBIAS', 'NSE_Q95', 'RMSE_Q95', 'UNDER_Q95', 'Peak_RMSE', 'Peak_Bias', 'Peak_Timing']
tt = C.test_table(met_all, EXPS, KEYS); display(tt); tt.to_csv(OUT / 'test_chi_tieu.csv', encoding='utf-8-sig')
tn = met_all.groupby(['model', 'exp']).NSE.mean().unstack()[EXPS].rename(index=DISPLAY).loc[[DISPLAY[m] for m in K.MODELS]]
print('NSE test (TB hạt giống):'); display(tn.round(3)); tn.to_csv(OUT / 'test_nse.csv', encoding='utf-8-sig')"""),
     md(r"""## 5. Bootstrap khối (tập kiểm tra, cùng một bộ chỉ số cho mọi cặp)

KTC 95 % không chứa 0 → có ý nghĩa (✔). P(A tốt hơn) = tỉ lệ lần bootstrap A tốt hơn B."""),
     code(r"""pred_all = pd.concat([pred0[pred0.exp.isin(['E2', 'FUT']) & pred0.model.isin(K.MODELS) & (pred0.h == 1)], predk])
pairs = []
for m in K.MODELS:
    pairs += [(m, 'KH_TD', m, 'E2', f'{DISPLAY[m]}: KH_TD − E2'), (m, 'KH_TD', m, 'FUT', f'{DISPLAY[m]}: KH_TD − FUT')]
    pairs += [(m, K.tag(k, s), m, 'E2', f'{DISPLAY[m]}: {K.tag(k, s)} − E2') for k in ['AD', 'SS'] for s in K.SIGMAS]
pairs += [(m, 'KH_TD', 'ridge', 'KH_TD', f'{DISPLAY[m]} − Ridge-ARX (KH_TD)') for m in ['lstm', 'gru']]
pairs += [(m, 'E2', 'ridge', 'E2', f'{DISPLAY[m]} − Ridge-ARX (E2, đối chứng)') for m in ['lstm', 'gru']]
bk = C.bootstrap_multi(cfg, pred_all, pairs); display(bk); bk.to_csv(OUT / 'bootstrap.csv', encoding='utf-8-sig')"""),
     md(r"""## 6. Mức sai số hoà vốn

σ lớn nhất mà biến thể còn hơn E2 có ý nghĩa về NSE (KTC 95 % > 0). "—" nghĩa là không mức nào có ý nghĩa."""),
     code(r"""rows = []
for m in K.MODELS:
    r = {'mô hình': DISPLAY[m]}
    for k in ['AD', 'SS']:
        ok = [s for s in K.SIGMAS if bk.loc[f'{DISPLAY[m]}: {K.tag(k, s)} − E2', 'KTC NSE'].endswith('✔')
              and bk.loc[f'{DISPLAY[m]}: {K.tag(k, s)} − E2', 'ΔNSE'] > 0]
        r[f'KH_{k}: σ lớn nhất còn hơn E2 có ý nghĩa'] = f'{int(max(ok) * 100)} %' if ok else '—'
    r['KH_TD − E2 có ý nghĩa'] = '✔' if bk.loc[f'{DISPLAY[m]}: KH_TD − E2', 'KTC NSE'].endswith('✔') else ''
    rows.append(r)
hv = pd.DataFrame(rows).set_index('mô hình'); display(hv); hv.to_csv(OUT / 'muc_sai_so_hoa_von.csv', encoding='utf-8-sig')"""),
     md(r"""## 7. Lợi ích theo mùa và trạng thái dòng chảy (MAE, mô tả, không kiểm định)

Như notebook 07, mục 12: nhóm theo **ngày đích**, ngưỡng Q50/Q90 từ 2019–2023, dự báo trung bình hạt giống. Hai nguồn: val
của CV (3 fold gộp) và test."""),
     code(r"""daily_ = pd.read_csv(ROOT / cfg['paths']['processed'] / 'cuadat_huana_mua_daily.csv', parse_dates=['date'], index_col='date')
q_ = daily_.q_cd[(daily_.index <= '2023-12-31') & ~daily_.cd_missing]; Q50, Q90 = q_.quantile(.5), q_.quantile(.9)
def groups(dt, o):
    fl = dt.month.isin([7, 8, 9, 10])
    return {'cả năm': np.ones(len(o), bool), 'mùa lũ T7–T10': fl, 'mùa kiệt T11–T6': ~fl,
            f'Q thấp (< {Q50:.0f})': o < Q50, f'Q trung bình ({Q50:.0f}–{Q90:.0f})': (o >= Q50) & (o < Q90), f'Q cao (≥ {Q90:.0f})': o >= Q90}
cvp0 = S.read_pred(cfg, 'cv', 1, 'chinh', K.MODELS, 'E2')
G_EX = ['E2', 'KH_TD', 'KH_AD20', 'KH_SS20']
rows = []
for src_, data in [('val CV (3 fold)', pd.concat([cvp0, cvpk])), ('test', pred_all)]:
    for m in K.MODELS:
        for e in G_EX:
            d = data[(data.model == m) & (data.exp == e) & (data.h == 1) & data.dich_that.astype(bool)]
            keys = ['fold', 'ngay_goc'] if 'fold' in d and d['fold'].notna().all() else ['ngay_goc']
            a = d.groupby(keys).agg(o=('Q_do', 'first'), p=('Q_du_bao', 'mean')).reset_index()
            dt = pd.DatetimeIndex(a.ngay_goc) + pd.Timedelta(days=1)
            for g, mk in groups(dt, a.o.values).items():
                rows.append({'nguồn': src_, 'mô hình': DISPLAY[m], 'thực nghiệm': e, 'nhóm': g, 'n': int(mk.sum()),
                             'MAE': float(np.abs(a.p.values[mk] - a.o.values[mk]).mean())})
hs = pd.DataFrame(rows)
piv = hs.pivot_table(index=['nguồn', 'nhóm', 'mô hình'], columns='thực nghiệm', values='MAE', sort=False)[G_EX]
for e in G_EX[1:]: piv[f'Δ% {e}'] = 100 * (piv[e] / piv['E2'] - 1)
piv['n'] = hs[hs['thực nghiệm'] == 'E2'].set_index(['nguồn', 'nhóm', 'mô hình']).n
piv.to_csv(OUT / 'theo_mua_trang_thai.csv', encoding='utf-8-sig')
med = piv.groupby(level=['nguồn', 'nhóm'], sort=False)[[f'Δ% {e}' for e in G_EX[1:]]].median()
print('Δ% MAE so với E2, trung vị 6 mô hình:'); display(med.round(1)); med.to_csv(OUT / 'theo_mua_trang_thai_trung_vi.csv', encoding='utf-8-sig')"""),
     md(r"""## 8. Đỉnh lũ trên tập kiểm tra (mô tả)"""),
     code(r"""pk = C.peak_table(pred_all, peaks, [(m, e) for m in ['ridge', 'lstm', 'gru'] for e in ['E2', 'KH_TD', 'KH_AD20']])
display(pk.round(1)); pk.to_csv(OUT / 'dinh_lu.csv', index=False, encoding='utf-8-sig')"""),
     md(r"""## 9. Hình"""),
     code(r"""fig, axs = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
xs = [0] + [int(s * 100) for s in K.SIGMAS]
cols = {'ridge': 'C2', 'extra_trees': 'C4', 'xgboost': 'C1', 'rnn': 'C5', 'lstm': 'C3', 'gru': 'C0'}
for ax, k, ttl in zip(axs, ['AD', 'SS'], ['(a) Huấn luyện bằng Q ra thực đo, dự báo bằng kế hoạch sai số σ (KH_AD)',
                                          '(b) Huấn luyện và dự báo bằng kế hoạch sai số σ (KH_SS)']):
    for m in K.MODELS:
        ys, lo, hi = [], [], []
        for e in ['KH_TD'] + [K.tag(k, s) for s in K.SIGMAS]:
            r = bk.loc[f'{DISPLAY[m]}: {e} − E2']
            ys.append(r['ΔNSE']); a_, b_ = r['KTC NSE'].strip(' ✔').strip('[]').split(';'); lo.append(float(a_)); hi.append(float(b_))
        ax.plot(xs, ys, marker='o', color=cols[m], label=DISPLAY[m])
        if m in ('ridge', 'lstm'): ax.fill_between(xs, lo, hi, color=cols[m], alpha=.12)
    ax.axhline(0, color='k', lw=.8); ax.set_xlabel('sai số kế hoạch σ (%)'); ax.set_title(ttl, loc='left', fontsize=9)
axs[0].set_ylabel('ΔNSE test so với E2'); axs[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / 'fig_delta_nse_theo_sai_so.png'); plt.show()

a, b = pd.Timestamp('2025-07-01'), pd.Timestamp('2025-10-31')
fig, axs = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
for ax, m in zip(axs, ['ridge', 'lstm']):
    for e, c in [('E2', 'C0'), ('KH_TD', 'C3'), ('KH_AD20', 'C1')]:
        O, M_, P, idx = C.arrays(pred_all, m, e); t = idx + pd.Timedelta(days=1)
        if e == 'E2': ax.plot(pd.Series(O[:, 0], index=t).loc[a:b], 'k', lw=1.3, label='quan trắc')
        ax.plot(pd.Series(np.mean(P, axis=0)[:, 0], index=t).loc[a:b], color=c, lw=1, label=e)
    ax.set_title(f'{DISPLAY[m]}: mùa lũ 2025, dự báo t+1 (TB hạt giống)', loc='left', fontsize=10); ax.set_ylabel('Q đến CĐ (m³/s)'); ax.legend(fontsize=8, ncol=4)
plt.tight_layout(); plt.savefig(OUT / 'fig_mua_lu_2025.png'); plt.show()""")]
save('09_KeHoachXa_HuaNa.ipynb', C)


# =====================================================================================================================
# 10 — chạy riêng một mô hình × một kịch bản (tương tác, KHÔNG ghi đè kết quả chính)
C = [md(r"""# 10. Chạy riêng một mô hình × một kịch bản: xem kết quả, chỉ test, hoặc huấn luyện từ đầu tới test

**Cách dùng:** sửa ô **0. Tham số** (mô hình, kịch bản, bài toán) rồi chọn **Run All**, hoặc chạy từng mục:

| Mục | Việc | Thời gian (RTX 5090) |
|---|---|---|
| 1 | Xem kết quả đã lưu của mô hình, mọi kịch bản (đọc `results/ket_qua_thuc_nghiem.csv`) | tức thì |
| 2 | **Chỉ test:** nạp weight tốt nhất đã lưu (E2, t+1), dự báo lại tập kiểm tra | vài giây |
| 3 | **Huấn luyện từ đầu tới test:** CV 3 fold → huấn luyện cuối 2019–2024 → chấm tập kiểm tra, kèm hình | 5–20 s (LSTM, GRU) |
| 4 | (tuỳ chọn) So sánh nhiều kịch bản của mô hình chính | 1–3 phút |

- Mọi bước dùng **đúng** quy trình của kết quả chính (`src/pipeline.py`): cùng dữ liệu đóng băng v1, cùng siêu tham số, cùng hạt giống. Trên cùng GPU, kết quả trùng với kết quả chính; mục 3 tự đối chiếu và báo "trùng".
- Notebook này **không ghi file nào**: kết quả hiện ngay trong notebook. Kết quả chính sinh bằng notebook 03–06 và 09.
- Lõi ở `src/run_single.py`. Bản dòng lệnh tương đương: `experiments/chay_mo_hinh.py`.

**Các kịch bản (`EXP`):**

| Mã | Nội dung | Mô hình |
|---|---|---|
| `PERS` | Q(t+1) = Q(t) | chỉ Persistence |
| `E1` | không có dữ liệu Hủa Na | 6 mô hình |
| `E2` | + Q ra Hủa Na (**cấu hình chính**) | 6 mô hình |
| `E3` | + Q đến Hủa Na | 6 mô hình |
| `E4` | + Q ra, Q đến, mực nước Hủa Na | 6 mô hình |
| `P_F4`, `P_F5`, `P_EMA` | thực nghiệm phụ | 6 mô hình |
| `E5` | E2 + hàm mất mát có trọng số theo lưu lượng | 6 mô hình |
| `E6`, `E56` | E2 + phạt dự báo thiếu (± trọng số) | RNN, LSTM, GRU |
| `FUT` | E2 + mưa thực đo t+1 (cận trên) | 6 mô hình |
| `KH_TD` | E2 + Q ra Hủa Na t+1 thực đo (cận trên) | 6 mô hình, chỉ H = 1 |
| `KH_AD10` … `KH_AD50` | mô hình KH_TD, dự báo bằng kế hoạch xả sai số 10–50 % | 6 mô hình, chỉ H = 1 |
| `KH_SS10` … `KH_SS50` | huấn luyện và dự báo bằng kế hoạch xả sai số 10–50 % | 6 mô hình, chỉ H = 1 |"""),
     code(SETUP + r"""
from src import run_single as R
from src.training.train import DISPLAY
from src.evaluation.metrics import per_horizon
pd.set_option('display.max_rows', 200)"""),
     md(r"""## 0. Tham số"""),
     code(r"""MODEL  = 'lstm'   # 'persistence', 'ridge', 'extra_trees', 'xgboost', 'rnn', 'lstm', 'gru'
EXP    = 'E2'     # mã kịch bản, xem bảng ở đầu notebook
H      = 1        # 1: bài toán chính t+1; 10: bài toán mở rộng t+1 … t+10
TU_DAU = False    # True: dò lại siêu tham số (và γ, λ) bằng CV thay vì đọc config.json — khoảng 2 phút với LSTM
print(f'{DISPLAY[MODEL]} · {EXP}: {R.describe(EXP)} · H = {H}')"""),
     md(r"""## 1. Kết quả đã lưu (không tính lại)

NSE CV là tiêu chí lựa chọn (trung bình 3 fold). Các cột còn lại là trung bình các hạt giống trên tập kiểm tra
01/2025 – 08/2026. ĐLC NSE: độ lệch chuẩn giữa các hạt giống."""),
     code(r"""display(R.saved_results(cfg, MODEL, H=H).round(3))"""),
     md(r"""## 2. Chỉ test: nạp mô hình cuối đã lưu và dự báo lại tập kiểm tra

Weight tốt nhất được lưu cho **E2, t+1** (5 hạt giống mỗi mô hình; Ridge-ARX tất định nên 1). Chạy trên CPU, nên dự
báo của RNN, LSTM, GRU có thể lệch dưới 1 m³/s so với bản đã lưu (tạo trên GPU); chỉ tiêu trùng tới 3 chữ số. Kịch bản
khác: xem mục 1 hoặc huấn luyện lại ở mục 3."""),
     code(r"""try:
    t_saved, pred_saved = R.test_saved(cfg, MODEL, EXP, H)
    display(t_saved[R.SHOW + ['chênh so với dự báo đã lưu (m³/s)']].round(3))
    print('TB:', ' · '.join(f'{k} {t_saved[k].mean():.3f}' for k in ['NSE', 'KGE', 'RMSE', 'NSE_Q95']))
except FileNotFoundError as e:
    print(e)"""),
     md(r"""## 3. Huấn luyện từ đầu tới test

1. CV 3 fold (huấn luyện 2019–2020 / 2019–2021 / 2019–2023; xác thực 2021 / 2022 / 2024), dừng sớm theo tập xác thực.
2. Số epoch (số cây với XGBoost) = trung vị số vòng tốt nhất trên CV.
3. Huấn luyện cuối trên 2019–2024 với 5 hạt giống, dự báo tập kiểm tra 01/2025 – 08/2026."""),
     code(r"""r = R.train_one(cfg, MODEL, EXP, H, TU_DAU)"""),
     md(r"""### 3.1 Kiểm định chéo theo fold (trung bình các hạt giống)"""),
     code(r"""display(r['cv'].groupby('fold')[['NSE', 'KGE', 'RMSE', 'MAE', 'PBIAS', 'NSE_Q95', 'best_iter']].mean().round(3))
print(f"NSE CV = {r['cv_tom_tat']['NSE CV']:.3f} (ĐLC giữa fold {r['cv_tom_tat']['NSE CV (ĐLC giữa fold)']:.3f})")"""),
     md(r"""### 3.2 Tập kiểm tra theo hạt giống"""),
     code(r"""t = r['test'].set_index('seed')[R.SHOW + ['n_iter']]
display(pd.concat([t, t.agg(['mean', 'std'])]).round(3))
if r['so_voi_ket_qua_chinh'] is not None:
    print(f"Đối chiếu với kết quả chính: chênh NSE lớn nhất {r['so_voi_ket_qua_chinh']:.1e}")"""),
     md(r"""### 3.3 Hình: dự báo t+1 trên tập kiểm tra (trung bình các hạt giống)"""),
     code(r"""o, p = R.mean_series(r['pred'], h=1)
fig = plt.figure(figsize=(15, 8)); gs = fig.add_gridspec(2, 3)
ax = fig.add_subplot(gs[0, :]); ax.plot(o, 'k', lw=1, label='quan trắc'); ax.plot(p, 'C3', lw=1, label=f'{DISPLAY[MODEL]} {EXP}')
ax.set_ylabel('Q đến Cửa Đạt (m³/s)'); ax.set_title('(a) Toàn bộ tập kiểm tra', loc='left'); ax.legend()
ax = fig.add_subplot(gs[1, :2]); a_, b_ = '2025-07-01', '2025-10-31'
ax.plot(o[a_:b_], 'k', lw=1.3, label='quan trắc'); ax.plot(p[a_:b_], 'C3', lw=1.2, label='dự báo')
ax.set_title('(b) Mùa lũ 2025', loc='left'); ax.legend()
ax = fig.add_subplot(gs[1, 2]); m_ = o.notna()
ax.scatter(o[m_], p[m_], s=6, alpha=.5); lim = [max(1, o.min()), o.max() * 1.1]
ax.plot(lim, lim, 'k--', lw=.8); ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('quan trắc'); ax.set_ylabel('dự báo')
ax.set_title('(c) Dự báo – quan trắc (thang log)', loc='left')
plt.tight_layout(); plt.show()
if H > 1:                                           # bài toán t+1 … t+10: chỉ tiêu theo bước dự báo
    rows = []
    for s, d in r['pred'].groupby('seed'):
        w = lambda c: d.pivot(index='ngay_goc', columns='h', values=c)
        rows.append(per_horizon(w('Q_do').values, w('Q_du_bao').values, w('dich_that').astype(bool).values)[['NSE', 'MAE']])
    ph = pd.concat(rows).groupby(level=0).mean(); display(ph.round(3).T)
    ph.plot(subplots=True, marker='o', figsize=(10, 3.5), layout=(1, 2), title=['NSE theo h', 'MAE theo h']); plt.show()"""),
     md(r"""## 4. (Tuỳ chọn) So sánh nhiều kịch bản của mô hình chính

Đặt `CHAY_LAI = True` để huấn luyện lại từng kịch bản (1–3 phút cho 2 mô hình × 8 kịch bản). Mặc định chỉ đọc kết quả đã
lưu."""),
     code(r"""MO_HINH  = ['lstm', 'gru']
KICH_BAN = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'FUT', 'KH_TD']
CHAY_LAI = False

if CHAY_LAI:
    rows = []
    for m in MO_HINH:
        for e in KICH_BAN:
            x = R.train_one(cfg, m, e, 1, log=lambda s: None)
            rows.append({'mô hình': DISPLAY[m], 'kịch bản': e, 'NSE CV': x['cv_tom_tat']['NSE CV'],
                         'NSE test': x['test'].NSE.mean(), 'ĐLC NSE test': x['test'].NSE.std(ddof=1),
                         'NSE Q>Q95 test': x['test'].NSE_Q95.mean(), 'Peak Bias (%)': x['test'].Peak_Bias.mean(),
                         'chênh so với kết quả chính': x['so_voi_ket_qua_chinh']})
            print(f'{DISPLAY[m]} {e}: xong')
    display(pd.DataFrame(rows).set_index(['mô hình', 'kịch bản']).round(3))
else:
    t = pd.concat([R.saved_results(cfg, m) for m in MO_HINH])
    display(t[t.index.get_level_values('kịch bản').isin(KICH_BAN)].round(3))""")]
save('10_ChayRieng_KichBan.ipynb', C)
