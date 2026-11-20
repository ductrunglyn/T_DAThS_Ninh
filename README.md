# Dự báo lưu lượng đến hồ Cửa Đạt có xét vận hành hồ Hủa Na

Mã thực nghiệm cho luận văn "Ứng dụng học sâu trong dự báo dòng chảy đến hồ Cửa Đạt có xét vận hành hồ Hủa Na".

Bảy mô hình được đánh giá trên **cùng** một bộ dữ liệu đóng băng, cùng cách tiền xử lý nhân quả, cùng ngày đích và cùng bộ chỉ tiêu:
- Persistence, Ridge-ARX, Extra Trees, XGBoost;
- RNN, LSTM, GRU (LSTM và GRU là mô hình chính).

Bài toán chính: dự báo Q đến Cửa Đạt ngày **t+1** từ thông tin đến hết ngày t. Bài toán mở rộng: t+1 … t+10.

**Code, dữ liệu và kết quả nằm ở hai nơi:**

| Nơi | Nội dung |
|---|---|
| GitHub [`ductrunglyn/T_DAThS_Ninh`](https://github.com/ductrunglyn/T_DAThS_Ninh) | Chỉ có code: các notebook, `src/`, `experiments/`, `tests/`. Notebook trên GitHub không kèm đầu ra. |
| Google Drive, thư mục [`T_DAThS_Ninh`](https://drive.google.com/drive/folders/1FOhh-bfsRCRbNZo3ljWV6eK_sTkj5zXc) | `Ninh_data.zip` (thư mục `data/`) và `Ninh_results.zip` (thư mục `results/`: kết quả và weight đã huấn luyện). |

- Kết quả sinh lại được bằng cách chạy notebook 03–09.
- Dữ liệu gốc `data/raw/` không sinh lại được, nên bắt buộc phải có `Ninh_data.zip`.
- Danh sách những gì không đưa lên GitHub nằm trong `.gitignore`.

**Tài liệu** (chỉ có trên máy, không đưa lên GitHub):
- báo cáo kết quả `docs/BaoCao_ThucNghiem.md`;
- đề cương luận văn kèm mô tả và phân tích dữ liệu `docs/DeCuong_ChiTiet_LuanVan.md`.

> **Cách làm việc chính là notebook**, đặt ngay ở thư mục gốc dự án (`01_….ipynb` … `10_….ipynb`).
> - Chạy trên Google Colab: mục 2.1. Chạy trên máy: mục 2.2.
> - Bản trên máy đã có sẵn đầu ra (bảng, hình): mở ra là xem được kết quả, không phải chạy lại.
> - Muốn chạy riêng một kịch bản, chỉ test, hoặc huấn luyện từ đầu tới test cho một mô hình, dùng **notebook 10** (mục 5.4).

---

## Mục lục

1. [Cấu trúc thư mục](#1-cấu-trúc-thư-mục)
   - [1.1 Thư mục kết quả](#11-thư-mục-kết-quả-results)
2. [Cài đặt và chạy](#2-cài-đặt-và-chạy)
   - [2.1 Google Colab](#21-google-colab)
   - [2.2 Máy cá nhân (conda)](#22-máy-cá-nhân-conda)
3. [Dữ liệu](#3-dữ-liệu)
4. [Các kịch bản thực nghiệm](#4-các-kịch-bản-thực-nghiệm)
5. [Làm việc bằng notebook](#5-làm-việc-bằng-notebook)
   - [5.1 Mở notebook và chọn kernel](#51-mở-notebook-và-chọn-kernel)
   - [5.2 Xem kết quả có sẵn (không cần chạy)](#52-xem-kết-quả-có-sẵn-không-cần-chạy)
   - [5.3 Chạy lại toàn bộ từ đầu](#53-chạy-lại-toàn-bộ-từ-đầu)
   - [5.4 Notebook 10: chạy riêng một kịch bản, chỉ test, huấn luyện từ đầu tới test](#54-notebook-10-chạy-riêng-một-kịch-bản-chỉ-test-huấn-luyện-từ-đầu-tới-test)
   - [5.5 Tự viết ô lệnh trong notebook](#55-tự-viết-ô-lệnh-trong-notebook)
6. [Chỉ test (không huấn luyện)](#6-chỉ-test-không-huấn-luyện)
7. [Huấn luyện từ đầu tới test: quy trình bên trong](#7-huấn-luyện-từ-đầu-tới-test-quy-trình-bên-trong)
8. [Chạy bằng dòng lệnh (tuỳ chọn)](#8-chạy-bằng-dòng-lệnh-tuỳ-chọn)
9. [Kiểm thử](#9-kiểm-thử)
10. [Cập nhật báo cáo sau khi chạy lại](#10-cập-nhật-báo-cáo-sau-khi-chạy-lại)
11. [Nguyên tắc và lưu ý](#11-nguyên-tắc-và-lưu-ý)
12. [Xử lý sự cố](#12-xử-lý-sự-cố)

---

## 1. Cấu trúc thư mục

```
01_ … 10_….ipynb          CÁC NOTEBOOK (cách làm việc chính), ở ngay thư mục gốc:
  01_PhanTich_XuLy_DuLieu   phân tích, làm sạch, kiểm tra chất lượng dữ liệu; mục 0: chuẩn bị trên Google Colab
  02_Dataset_ChiaTap        điền thiếu, đặc trưng, chia tập, đóng băng dữ liệu v1
  03_Baseline               Persistence, Ridge-ARX, Extra Trees, XGBoost
  04_RNN · 05_LSTM · 06_GRU mô hình học sâu (05, 06 là mô hình chính)
  07_SoSanh                 so sánh 7 mô hình, bootstrap, trả lời 3 câu hỏi nghiên cứu
  08_DieuTra_LSTM_E5        điều tra lần huấn luyện LSTM-E5 cho kết quả thấp
  09_KeHoachXa_HuaNa        kịch bản bổ sung KH: kế hoạch xả Hủa Na ngày t+1
  10_ChayRieng_KichBan      chạy riêng một mô hình × một kịch bản; chỉ test; huấn luyện từ đầu tới test
data/                     [Google Drive: Ninh_data.zip]
data/raw/                 dữ liệu gốc (.xlsx) và bản .csv chuyển đổi — không sửa
data/processed/           bảng ngày đã làm sạch, giữ NaN ở ngày thiếu (notebook 01)
data/frozen/v1/           bộ dữ liệu ĐÓNG BĂNG + manifest SHA-256; mọi mô hình chỉ đọc từ đây
src/data/                 làm sạch, điền thiếu nhân quả, đặc trưng, chuẩn hoá, chia tập, cửa sổ, đóng băng
src/models/               persistence · ridge_arx · extra_trees · xgboost_model · rnn · lstm · gru (recurrent.py dùng chung)
src/training/             huấn luyện thống nhất · hàm mất mát · kiểm định chéo và dò siêu tham số
src/evaluation/           chỉ tiêu · bootstrap · tổng hợp so sánh
src/pipeline.py           quy trình chung: dò siêu tham số → CV → huấn luyện cuối + test → bài toán 10 ngày
src/plan_scenario.py      kịch bản KH: ghép Q ra Hủa Na ngày t+1 (thực đo / kế hoạch có sai số) vào E2
src/run_single.py         chạy riêng một mô hình × một kịch bản (dùng trong notebook 10)
src/results_store.py      đọc/ghi các file kết quả gộp (mục 1.1)
experiments/config.yaml          cấu hình DUY NHẤT (dữ liệu, chia tập, lưới siêu tham số, hạt giống, bootstrap)
experiments/run_experiments.py   chạy các notebook theo thứ tự bằng dòng lệnh (tuỳ chọn)
experiments/build_notebooks.py   sinh nội dung notebook
experiments/build_docs.py        sinh phụ lục bảng, danh mục hình của báo cáo và phần dữ liệu của đề cương (chỉ trên máy)
tests/                    chống rò rỉ dữ liệu · nhất quán dữ liệu · tái lập mô hình đã lưu · kịch bản KH (59 test)
results/                  KẾT QUẢ (mục 1.1): 4 file gộp + weight tốt nhất + bảng phân tích, hình   [Google Drive: Ninh_results.zip]
docs/                     báo cáo, đề cương, kiểm tra và xác minh dữ liệu, từ điển dữ liệu, thiết kế kịch bản KH (chỉ trên máy)
requirements.txt          thư viện cho máy cá nhân (mục 2.2)
requirements_colab.txt    thư viện cài thêm trên Google Colab (ô đầu tiên của notebook tự cài)
.gitignore                những gì KHÔNG đưa lên GitHub: data/, results/, docs/, build_docs.py, file .zip
```

### 1.1 Thư mục kết quả `results/`

```
results/
  ket_qua_thuc_nghiem.csv    MỌI chỉ tiêu: CV (theo fold × hạt giống) và test (theo hạt giống), mọi mô hình × kịch bản
  du_bao.csv.gz              MỌI dự báo theo ngày (CV và test) — để tính lại bootstrap, vẽ hình, kiểm thử không cần huấn luyện lại
  do_sieu_tham_so.csv        lưới dò siêu tham số của mọi mô hình và NSE CV từng cấu hình
  cau_hinh_chon.json         cấu hình chọn theo CV: siêu tham số, γ của E5, λ của E6 (thí nghiệm B và A)
  baselines/<mô hình>/final_models/, models/<mô hình>/final_models/
                             WEIGHT TỐT NHẤT: cấu hình chính E2, t+1, 5 hạt giống (Ridge-ARX tất định: 1) + mo_hinh.json (mô tả)
  models/<rnn|lstm|gru>/train_log_H1.csv, train_log_H10.csv   nhật ký huấn luyện theo epoch
  models/lstm/dieu_tra_E5/   điều tra lần huấn luyện LSTM-E5 (notebook 08)
  comparison/                bảng so sánh, bootstrap, hình (notebook 07)
  ke_hoach_xa/               bảng phân tích, bootstrap, hình của kịch bản KH (notebook 09)
  phan_tich_du_lieu/, dataset/   phân tích dữ liệu, chia tập, kiểm tra thiết kế (notebook 01, 02)
```

**`ket_qua_thuc_nghiem.csv`** có một dòng cho mỗi (nhóm, bài toán, giai đoạn, mô hình, kịch bản, fold, hạt giống):

| Cột | Ý nghĩa |
|---|---|
| `nhom` | `chinh` (kết quả chính, thí nghiệm B) · `A_L3` (thí nghiệm A, cửa sổ chung L = 3; chỉ Ridge-ARX khác B) · `KH` (kịch bản kế hoạch xả) |
| `H` | `1` (t+1) · `10` (t+1 … t+10; chỉ tiêu là trung bình 10 bước) |
| `giai_doan`, `fold` | `cv` với fold `F1`, `F2`, `F3` (xác thực 2021, 2022, 2024) · `test` với fold `FINAL` (tập kiểm tra 2025–2026) |
| `model`, `exp`, `seed` | mô hình, mã kịch bản (mục 4), hạt giống |
| `so_vong`, `time_s` | số epoch / số cây (CV: tốt nhất theo dừng sớm; test: số dùng khi huấn luyện cuối) · thời gian huấn luyện (giây, chỉ CV) |
| `MSE` … `Peak_Timing` | chỉ tiêu: MSE, MAE, RMSE, r², NSE, KGE, PBIAS, dự báo thiếu TB, NSE / RMSE / dự báo thiếu khi Q > Q95, Peak RMSE, Peak Bias, sai số thời điểm đỉnh |
| `fs`, `gamma`, `cap`, `lam`, `fut` | tập biến, trọng số E5 (γ, chặn trên), λ của E6, có mưa tương lai (FUT) |

**`du_bao.csv.gz`** có các cột `nhom, H, giai_doan, model, exp, fold, seed, ngay_goc, h, Q_do, dich_that, Q_du_bao`
(ngày gốc t, bước h, Q quan trắc tại t+h, có quan trắc thật không, Q dự báo).

Đọc nhanh trong notebook:

```python
import pandas as pd
kq = pd.read_csv('results/ket_qua_thuc_nghiem.csv')
kq[(kq.giai_doan == 'test') & (kq.H == 1) & (kq.nhom == 'chinh')].groupby(['model', 'exp']).NSE.agg(['mean', 'std'])

from src import results_store as S                        # hoặc dùng hàm đọc có sẵn (cùng định dạng các bảng trong code)
S.read_metrics(cfg, 'test', H=1, nhom='chinh', model='lstm')
S.read_pred(cfg, 'test', H=1, nhom='chinh', model='lstm', exp='E2')
```

Mọi lần chạy ghi vào các file gộp theo kiểu **thay thế**: chạy lại một mô hình chỉ thay các dòng của mô hình đó, không
làm trùng dòng.

---

## 2. Cài đặt và chạy

### 2.1 Google Colab

Trên Colab, code lấy từ GitHub, còn dữ liệu và kết quả giải nén từ file `.zip`. Cả hai nằm trong **cùng một thư mục trên
Drive**: `MyDrive/T_DAThS_Ninh/`, cũng là thư mục chứa hai file `.zip`. Kết quả vì vậy không mất khi tắt Colab, và mọi
notebook dùng chung thư mục này.

**Chuẩn bị (làm một lần):**

1. Trong My Drive có thư mục `T_DAThS_Ninh` ([link](https://drive.google.com/drive/folders/1FOhh-bfsRCRbNZo3ljWV6eK_sTkj5zXc)),
   chứa hai file nén (giữ nguyên tên file):

   | File | Nội dung | Bắt buộc |
   |---|---|---|
   | `Ninh_data.zip` (~1 MB) | thư mục `data/`: dữ liệu gốc, bảng ngày, dữ liệu đóng băng v1 | có |
   | `Ninh_results.zip` (~110 MB) | thư mục `results/`: 4 file kết quả gộp, weight tốt nhất, bảng, hình | không: thiếu thì chạy notebook 03–09 để sinh lại |

2. Mở notebook 01 trên Colab, bằng một trong hai cách:
   - link trực tiếp: [01_PhanTich_XuLy_DuLieu.ipynb trên Colab](https://colab.research.google.com/github/ductrunglyn/T_DAThS_Ninh/blob/main/01_PhanTich_XuLy_DuLieu.ipynb);
   - hoặc https://colab.research.google.com → **File → Open notebook → GitHub** → nhập `ductrunglyn/T_DAThS_Ninh` → chọn
     `01_PhanTich_XuLy_DuLieu.ipynb`.
3. **Runtime → Change runtime type → T4 GPU.**
4. Chạy hai ô của **mục 0**:
   - **ô 0a**: gắn Google Drive (Colab hỏi quyền truy cập, chọn cho phép), rồi lấy code từ GitHub vào
     `MyDrive/T_DAThS_Ninh/`, cạnh hai file `.zip`;
   - **ô 0b**: giải nén `Ninh_data.zip` và `Ninh_results.zip` ngay trong thư mục đó, tạo ra `data/` và `results/`.
   - Nếu thư mục `T_DAThS_Ninh` không nằm ngay trong My Drive, ô 0a tự tìm sâu hơn (tối đa 3 cấp).
5. Chạy tiếp phần còn lại của notebook 01.

**Các lần sau và các notebook khác:**
- Mở notebook bằng một trong hai cách:
  - từ Drive: `MyDrive/T_DAThS_Ninh/` → nhấp đúp file `.ipynb` → **Open with Google Colaboratory**;
  - từ GitHub như bước 2.
- Ô đầu tiên của mọi notebook tự làm ba việc, không phải sửa gì:
  - gắn Drive;
  - vào `MyDrive/T_DAThS_Ninh/`;
  - cài đúng phiên bản thư viện (`requirements_colab.txt`).
- Mỗi notebook trên Colab chạy trên một máy riêng, nên ô đầu tiên lặp lại việc này ở mọi notebook (khoảng 30 giây).
- Kết quả ghi thẳng vào `MyDrive/T_DAThS_Ninh/results/`.
- Thứ tự chạy và nội dung từng notebook như trên máy (mục 5).

**Lưu ý:**
- **Thư viện.** Colab có sẵn torch, numpy, pandas. Chỉ cài thêm scikit-learn 1.7.2 và xgboost 3.2.0 (đúng bản đã dùng) để nạp được
  weight Ridge-ARX, Extra Trees, XGBoost đã lưu.
- **Sai lệch trên phần cứng khác** (kết quả đã lưu chạy trên RTX 5090):
  - Persistence, Ridge-ARX, Extra Trees, XGBoost cho kết quả như nhau;
  - chỉ test bằng weight đã lưu (notebook 10, mục 2): chỉ tiêu trùng tới 3 chữ số;
  - huấn luyện lại RNN, LSTM, GRU thì lệch: thử LSTM-E2 trên CPU được NSE trung bình 0,517 (đã lưu 0,520), từng hạt giống
    lệch tới 0,02. Chạy lại cả notebook 04–06 thì siêu tham số chọn theo CV cũng có thể khác;
  - notebook 02 có thể chọn ngưỡng G khác (trên CPU chọn G = 7), nhưng vẫn giữ bộ dữ liệu đóng băng v1 (G = 3) và in cảnh báo.
  - Số liệu của luận văn là kết quả trong `Ninh_results.zip`.
- **Ghi đè kết quả.** Chạy lại notebook 03–09 sẽ ghi đè `results/` trên Drive. Muốn quay về kết quả gốc: đặt `GHI_DE = True` ở
  ô 0b rồi chạy lại ô đó.
- **Cập nhật code.** Chạy lại ô 0a để lấy code mới nhất từ GitHub (`git pull`).
- **Đặt dự án chỗ khác trên Drive.** Sửa `DU_AN` ở ô 0a và ở ô đầu tiên của các notebook. Ô đầu tiên cũng tự tìm thư mục dự
  án trong My Drive, sâu tối đa 3 cấp.
- **Thư mục được chia sẻ từ tài khoản khác.** Chuột phải vào thư mục → **Organize → Add shortcut → My Drive**, để nó xuất
  hiện trong `MyDrive/`.

### 2.2 Máy cá nhân (conda)

Các thực nghiệm được chạy với:

| Thành phần | Phiên bản |
|---|---|
| Hệ điều hành | Linux |
| Môi trường | conda env `MTL`, Python 3.10.20 |
| GPU | RTX 5090, CUDA 12.8 |
| Thư viện chính | torch 2.8.0 + cu128, scikit-learn 1.7.2, xgboost 3.2.0, pandas 2.3.2, numpy 1.26.4 |

**Bước 0. Lấy code và dữ liệu** (bỏ qua nếu đã có thư mục dự án)

```bash
git clone https://github.com/ductrunglyn/T_DAThS_Ninh.git
cd T_DAThS_Ninh
unzip Ninh_data.zip          # tải từ Google Drive về thư mục này; tạo ra data/
unzip Ninh_results.zip       # không bắt buộc; tạo ra results/
```

**Bước 1. Tạo môi trường**

```bash
conda create -n MTL python=3.10 -y
conda activate MTL
```

**Bước 2. Cài PyTorch đúng bản CUDA của máy** (xem https://pytorch.org). Với CUDA 12.8:

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

Máy không có GPU dùng bản CPU: `pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu`.

**Bước 3. Cài các thư viện còn lại.** File `requirements.txt` có sẵn `ipykernel`, đủ để chạy notebook trong VS Code.

```bash
cd BacSon/Ninh                      # thư mục gốc dự án (chứa các notebook, src/, data/)
pip install -r requirements.txt
```

**Bước 4. Công cụ mở notebook** (chọn một trong hai):
- **VS Code** (khuyến nghị): cài hai extension **Python** và **Jupyter** của Microsoft. Không cần cài thêm gì trong môi trường.
- **Jupyter Lab**: `pip install jupyterlab`. Môi trường `MTL` hiện chưa có gói này.

**Bước 5. Kiểm tra cài đặt**

```bash
python -c "import torch, sklearn, xgboost, pandas; print(torch.__version__, torch.cuda.is_available())"
python -m pytest -q tests/          # 59 passed
```

**Chạy không có GPU:**
- Mã tự dùng CPU khi không có GPU, chậm hơn vài lần.
- Ridge-ARX, Extra Trees, XGBoost cho kết quả như nhau.
- RNN, LSTM, GRU:
  - nạp weight đã lưu và dự báo: lệch dưới 1 m³/s;
  - huấn luyện lại: NSE từng hạt giống lệch tới khoảng 0,02 so với kết quả đã lưu (xem mục 2.1, phần lưu ý).

---

## 3. Dữ liệu

| File (`data/raw/`) | Nội dung |
|---|---|
| `vanhanh_cuadat_2019-2026.csv` (.xlsx) | Số liệu vận hành ngày hồ Cửa Đạt |
| `Vanhanh_huaNa_2019-2026.csv` (.xlsx) | Số liệu vận hành ngày hồ Hủa Na |
| `CHIRPS_FINAL_2VUNG_1986_2026_FORECAST.csv` | Mưa CHIRPS trung bình vùng: V1 (lưu vực Hủa Na), V2 (khu giữa) |

**Quy trình dữ liệu:**
- Notebook 01 làm sạch số liệu và hợp nhất thành bảng ngày `data/processed/cuadat_huana_mua_daily.csv`.
- Notebook 02 chọn ngưỡng điền thiếu G = 3, dựng các mẫu, chia tập, rồi **đóng băng** vào `data/frozen/v1/`. Manifest lưu SHA-256 của từng file, của phần cấu hình dữ liệu và của mã `src/data/`.
- Mọi mô hình chỉ đọc dữ liệu qua `pipeline.get_dataset()`. Hàm này kiểm tra manifest trước khi chạy, sai một byte là dừng.

**Chia tập** (cố định trong `config.yaml`):

| Tách | Huấn luyện | Xác thực / kiểm tra |
|---|---|---|
| F1 | 2019 – 2020 | xác thực 2021 |
| F2 | 2019 – 2021 | xác thực 2022 |
| F3 | 2019 – 2023 | xác thực 2024 |
| FINAL | 2019 – 2024 | **kiểm tra** 01/01/2025 – 11/08/2026 (ngày gốc) |

Tập kiểm tra chỉ dùng để báo cáo, không dùng cho bất kỳ lựa chọn nào.

---

## 4. Các kịch bản thực nghiệm

Mã kịch bản dùng chung trong notebook, file kết quả và biến `EXP` của notebook 10.

| Mã | Đầu vào / thay đổi | Mô hình | Câu hỏi nghiên cứu |
|---|---|---|---|
| `PERS` | Q(t+1) = Q(t) | Persistence | mốc tham chiếu |
| `E1` | F0: Q Cửa Đạt, mưa V1, V2, lịch, cờ thiếu (không có Hủa Na) | 6 mô hình | 2 |
| `E2` | F1 = F0 + Q ra Hủa Na (**cấu hình chính**) | 6 mô hình | 1, 2 |
| `E3` | F0 + Q đến Hủa Na | 6 mô hình | 2 |
| `E4` | F0 + Q ra + Q đến + mực nước Hủa Na | 6 mô hình | 2 |
| `P_F4`, `P_F5`, `P_EMA` | phụ: đủ 6 biến Hủa Na; E4 + đặc trưng thủy văn; E2 + trung bình trượt mũ | 6 mô hình | 2, 3 |
| `E5` | E2 + hàm mất mát có trọng số theo lưu lượng (γ, chặn trên chọn theo CV) | 6 mô hình | 3 |
| `E6` | E2 + phạt dự báo thiếu (λ chọn theo CV) | RNN, LSTM, GRU | 3 |
| `E56` | E2 + trọng số + phạt dự báo thiếu | RNN, LSTM, GRU | 3 |
| `FUT` | E2 + mưa **thực đo** ngày t+1 (cận trên, không khả dụng khi vận hành) | 6 mô hình | 3 |
| `KH_TD` | E2 + Q ra Hủa Na ngày t+1 thực đo (cận trên) | 6 mô hình | 2 (bổ sung) |
| `KH_AD10/20/30/50` | mô hình KH_TD, dự báo bằng kế hoạch xả sai số σ = 10/20/30/50 % | 6 mô hình | 2 (bổ sung) |
| `KH_SS10/20/30/50` | huấn luyện và dự báo bằng kế hoạch xả sai số σ | 6 mô hình | 2 (bổ sung) |

**Ghi chú:**
- "6 mô hình" là mọi mô hình trừ Persistence.
- **Bài toán t+1 … t+10:** kết quả chính có E1, E2, E5. Kịch bản KH chỉ làm cho t+1.
- **Thí nghiệm A** (mọi mô hình dùng chung cửa sổ L = 3) nằm trong notebook 03–06, mục "thí nghiệm A".
- Thiết kế kịch bản KH được viết trước khi chạy, ở [`docs/ThietKe_KichBan_KeHoachXa.md`](docs/ThietKe_KichBan_KeHoachXa.md).

---

## 5. Làm việc bằng notebook

### 5.1 Mở notebook và chọn kernel

**VS Code:**
1. Mở thư mục dự án `BacSon/Ninh` (File → Open Folder).
2. Mở một notebook ở thư mục gốc dự án, ví dụ `05_LSTM.ipynb`.
3. Bấm **Select Kernel** ở góc trên bên phải → **Python Environments…** → chọn **MTL** (`~/miniconda3/envs/MTL/bin/python`).
4. Chạy:
   - **Run All**: chạy cả notebook;
   - **Shift + Enter**: chạy từng ô;
   - **Clear All Outputs**: xoá đầu ra trên màn hình.

**Jupyter Lab:**

```bash
conda activate MTL
cd BacSon/Ninh
jupyter lab
```

Chọn kernel **Python 3** của môi trường MTL.

Ô đầu tiên của mọi notebook tự tìm thư mục gốc dự án và đọc `experiments/config.yaml`, nên notebook chạy được dù mở từ đâu. Ô này in ra `Kết quả lưu ở: results`.

### 5.2 Xem kết quả có sẵn (không cần chạy)

Bản trên máy: mọi notebook đã được chạy và **lưu sẵn đầu ra**. Chỉ cần mở file và kéo xuống, **không bấm Run**.
Bản trên GitHub không kèm đầu ra; trên Colab, mở notebook rồi chạy (cần `Ninh_results.zip` để có kết quả mà không phải huấn luyện):

| Muốn xem | Mở notebook | Mục |
|---|---|---|
| Phân tích dữ liệu: chuỗi thời gian, chất lượng, tương quan, độ trễ | `01_PhanTich_XuLy_DuLieu` | 1–4 |
| Chọn ngưỡng điền thiếu G, chia tập, kiểm tra thiết kế | `02_Dataset_ChiaTap` | 1–6 |
| LSTM: siêu tham số đã chọn, mọi kịch bản trên CV, kết quả test, thí nghiệm A, t+1 … t+10 | `05_LSTM` | theo thứ tự |
| GRU: như trên | `06_GRU` | theo thứ tự |
| RNN; các mô hình so sánh (Persistence, Ridge-ARX, Extra Trees, XGBoost) | `04_RNN`, `03_Baseline` | theo thứ tự |
| **Câu hỏi 1**: LSTM, GRU so với các mô hình khác (bảng test, bootstrap) | `07_SoSanh` | 1, 2, 8, 11 |
| **Câu hỏi 2**: vai trò dữ liệu Hủa Na (E1–E4), theo mùa và trạng thái dòng chảy | `07_SoSanh` | 3, 12 |
| **Câu hỏi 3**: điền thiếu, E5, E6, E56, P_EMA, FUT; độ ổn định hạt giống | `07_SoSanh` | 4, 9 |
| Bài toán t+1 … t+10 | `07_SoSanh` | 5 |
| Thí nghiệm A và B; kết quả theo fold | `07_SoSanh` | 6, 7 |
| Đỉnh lũ từng sự kiện, độ nhạy; chi phí huấn luyện | `07_SoSanh` | 9, 10 |
| Lần huấn luyện LSTM-E5 cho kết quả thấp | `08_DieuTra_LSTM_E5` | 1–5 |
| Kịch bản KH: kế hoạch xả Hủa Na | `09_KeHoachXa_HuaNa` | 3–9 |

Mọi bảng cũng được lưu thành `.csv` trong `results/`. Hình được lưu thành `.png` trong `results/` và chép vào `docs/hinh/`.

### 5.3 Chạy lại toàn bộ từ đầu

Mở và **Run All** lần lượt các notebook theo đúng thứ tự:

| Thứ tự | Notebook | Việc | Thời gian (RTX 5090) |
|---|---|---|---|
| 1 | `01_PhanTich_XuLy_DuLieu` | Làm sạch, hợp nhất, phân tích dữ liệu (không huấn luyện) | < 1 phút |
| 2 | `02_Dataset_ChiaTap` | Chọn G bằng CV, chia tập, đóng băng v1, chạy pytest | ~1 phút |
| 3 | `03_Baseline` | 4 mô hình so sánh | ~2 phút |
| 4 | `04_RNN` | RNN | ~3 phút |
| 5 | `05_LSTM` | **LSTM** | ~4 phút |
| 6 | `06_GRU` | **GRU** | ~3,5 phút |
| 7 | `07_SoSanh` | So sánh, bootstrap, hình (không huấn luyện) | ~1 phút |
| 8 | `08_DieuTra_LSTM_E5` | Điều tra LSTM-E5 (cần 05) | ~1 phút |
| 9 | `09_KeHoachXa_HuaNa` | Kịch bản KH (cần 03–06) | ~7 phút |

Mỗi notebook mô hình (03–06) làm theo cùng 5 bước, mỗi bước một mục:
1. Dò siêu tham số trên E2.
2. Chạy mọi kịch bản trên CV, chọn γ (E5) và λ (E6).
3. Huấn luyện cuối 2019–2024 và chấm tập kiểm tra.
4. Thí nghiệm A (L = 3).
5. Bài toán t+1 … t+10.

Mục 7 giải thích chi tiết từng bước.

> **Lưu ý.**
> - Chạy lại notebook 02–06, 08, 09 sẽ **ghi đè** kết quả chính trong `results/`. Trên cùng GPU, kết quả trùng hoàn toàn với lần chạy trước. Đã kiểm tra ngày 20/11/2026: chạy lại toàn bộ 01–10, mọi chỉ tiêu, dự báo, lưới siêu tham số và cấu hình chọn trùng bản trước.
> - Chỉ muốn thử một kịch bản mà không đụng kết quả chính thì dùng notebook 10.
> - Sau khi chạy lại 03–06 hoặc 09, nên chạy lại 07 để cập nhật bảng so sánh.

### 5.4 Notebook 10: chạy riêng một kịch bản, chỉ test, huấn luyện từ đầu tới test

Mở `10_ChayRieng_KichBan.ipynb`, sửa ô **0. Tham số**, rồi **Run All**:

```python
MODEL  = 'lstm'   # 'persistence', 'ridge', 'extra_trees', 'xgboost', 'rnn', 'lstm', 'gru'
EXP    = 'E2'     # mã kịch bản, xem bảng ở mục 4
H      = 1        # 1: bài toán chính t+1; 10: bài toán mở rộng t+1 … t+10
TU_DAU = False    # True: dò lại siêu tham số (và γ, λ) bằng CV thay vì đọc cấu hình đã chọn
```

| Mục | Việc | Thời gian |
|---|---|---|
| 1 | Xem kết quả đã lưu của `MODEL` (mọi kịch bản): NSE CV và chỉ tiêu tập kiểm tra | tức thì |
| 2 | **Chỉ test:** nạp weight tốt nhất đã lưu (E2, t+1), dự báo lại tập kiểm tra, tính chỉ tiêu | vài giây |
| 3 | **Huấn luyện từ đầu tới test:** CV 3 fold → huấn luyện cuối 2019–2024 → chấm tập kiểm tra. In kết quả theo fold, theo hạt giống, kèm hình chuỗi dự báo, mùa lũ 2025, biểu đồ phân tán; với H = 10 thêm NSE, MAE theo bước h | 5–20 s (LSTM, GRU) |
| 4 | (tuỳ chọn) Bảng so sánh nhiều kịch bản của LSTM, GRU. Mặc định đọc kết quả đã lưu; đặt `CHAY_LAI = True` để huấn luyện lại | 1–3 phút |

**Các điểm cần biết:**
- Mục 3 dùng **đúng** quy trình, dữ liệu, siêu tham số và hạt giống của kết quả chính, rồi tự đối chiếu với kết quả chính. Dòng `Đối chiếu với kết quả chính đã lưu: chênh NSE lớn nhất 0.00e+00 (trùng)` nghĩa là tái lập đúng.
- Notebook 10 **không ghi file nào**: kết quả hiện ngay trong notebook, kết quả chính không bị đụng tới.
- Chỉ muốn chạy một phần thì chạy ô thiết lập (đầu tiên), ô Tham số, rồi chạy riêng mục cần.
  - Ví dụ chỉ test: chạy mục 2.
  - Chạy mục 3 thì phải chạy ô `r = R.train_one(...)` trước các ô 3.1–3.3.

**Ví dụ tham số cho hai mô hình chính:**

| Mục đích | `MODEL` | `EXP` | `H` | `TU_DAU` |
|---|---|---|---|---|
| Cấu hình chính (câu hỏi 1) | `'lstm'` / `'gru'` | `'E2'` | 1 | False |
| Không có dữ liệu Hủa Na (câu hỏi 2) | `'lstm'` | `'E1'` | 1 | False |
| Q đến Hủa Na; đủ ba biến Hủa Na | `'lstm'` | `'E3'`; `'E4'` | 1 | False |
| Hàm mất mát có trọng số; phạt dự báo thiếu (câu hỏi 3) | `'gru'` | `'E5'`; `'E6'`; `'E56'` | 1 | False |
| Mưa thực đo t+1 (cận trên) | `'lstm'` | `'FUT'` | 1 | False |
| Kế hoạch xả Hủa Na t+1 thực đo; sai số 20 % | `'lstm'` | `'KH_TD'`; `'KH_AD20'` | 1 | False |
| Bài toán t+1 … t+10 | `'lstm'` | `'E2'` | 10 | False |
| Dò lại siêu tham số từ đầu (khoảng 2 phút) | `'lstm'` | `'E2'` | 1 | True |

**Chỉ test bằng mô hình đã lưu (mục 2):**
- Weight tốt nhất được lưu cho **cấu hình chính E2, t+1**, ở 6 mô hình, mỗi mô hình 5 hạt giống (Ridge-ARX tất định: 1). Kịch bản khác in thông báo; xem mục 1, hoặc huấn luyện lại ở mục 3 (vài giây).
- Mục 2 chạy trên CPU. Dự báo của RNN, LSTM, GRU có thể lệch dưới 1 m³/s so với bản đã lưu (tạo trên GPU); chỉ tiêu trùng tới 3 chữ số.

### 5.5 Tự viết ô lệnh trong notebook

Có thể thêm ô mới vào notebook bất kỳ (sau ô thiết lập đầu tiên) và gọi trực tiếp các hàm:

```python
from src import run_single as R

R.saved_results(cfg, 'gru')                       # bảng kết quả đã lưu của GRU
t, pred = R.test_saved(cfg, 'lstm', 'E4')         # chỉ test bằng mô hình đã lưu
r = R.train_one(cfg, 'lstm', 'E5')                # huấn luyện từ đầu tới test (không ghi file)
r['cv']                                           # chỉ tiêu CV theo fold × hạt giống
r['test']                                         # chỉ tiêu tập kiểm tra theo hạt giống
r['pred']                                         # dự báo tập kiểm tra
o, p = R.mean_series(r['pred'])                   # quan trắc và dự báo trung bình theo ngày đích, để vẽ
```

Chạy lại **toàn bộ quy trình của một mô hình**, như notebook 05. Cách này **ghi đè** kết quả chính của mô hình đó:

```python
from src import pipeline
tab, best = pipeline.tune(cfg, 'lstm')            # 1. dò siêu tham số
cvt = pipeline.cv_experiments(cfg, 'lstm')        # 2. mọi kịch bản trên CV
met = pipeline.final_test(cfg, 'lstm')            # 3. huấn luyện cuối + tập kiểm tra
met.groupby('exp').NSE.agg(['mean', 'std'])
```

---

## 6. Chỉ test (không huấn luyện)

| Mức | Cách | Có cho |
|---|---|---|
| Đọc lại kết quả test đã lưu | Notebook 10, mục 1; hoặc mở notebook 03–06, 09 | mọi mô hình × kịch bản × hạt giống |
| Nạp weight tốt nhất, dự báo lại tập kiểm tra | Notebook 10, mục 2 | E2, t+1 (6 mô hình) |
| Tính lại bootstrap, vẽ lại hình từ dự báo đã lưu (`results/du_bao.csv.gz`) | Chạy lại notebook 07 hoặc 09 từ mục 3 | mọi mô hình × kịch bản |
| Kiểm thử tái lập tự động | `python -m pytest -q tests/test_final_models.py` | weight đã lưu |

Muốn huấn luyện lại rồi test cho kịch bản không có mô hình đã lưu, dùng notebook 10, mục 3.

---

## 7. Huấn luyện từ đầu tới test: quy trình bên trong

Quy trình giống hệt cho mọi mô hình, mã ở `src/pipeline.py`. Mỗi bước tương ứng một mục trong notebook 03–06.

```
dữ liệu đóng băng v1
   │
   ├─ 1. Dò siêu tham số            pipeline.tune()            E2, t+1, CV 3 fold × 2 hạt giống; chọn theo NSE CV
   │                                                            → do_sieu_tham_so.csv, cau_hinh_chon.json
   ├─ 2. Thực nghiệm trên CV        pipeline.cv_experiments()  mọi kịch bản với siêu tham số đã chọn;
   │                                                            chọn γ (E5), λ (E6) theo NSE CV
   │                                                            → ket_qua_thuc_nghiem.csv, du_bao.csv.gz (giai_doan = cv)
   ├─ 3. Huấn luyện cuối + test     pipeline.final_test()      huấn luyện 2019–2024, 5 hạt giống,
   │                                                            số epoch / số cây = trung vị vòng tốt nhất ở bước 2;
   │                                                            → ket_qua_thuc_nghiem.csv, du_bao.csv.gz (giai_doan = test);
   │                                                            weight E2 → final_models/
   ├─ 4. Thí nghiệm A               pipeline.run_common_L()    như 1–3 nhưng cố định L = 3 → nhom = A_L3
   └─ 5. t+1 … t+10                 pipeline.extended()        E1, E2, E5, cùng siêu tham số → H = 10
```

**Chi tiết huấn luyện mạng hồi tiếp:**
- Kiến trúc: chuỗi L × F → RNN/LSTM/GRU (1–2 lớp, 32/64/128 nút ẩn) → trạng thái ẩn cuối → Dropout 0,2 → Linear.
- Tối ưu: Adam (lr 10⁻³), batch 64, cắt chuẩn gradient ở 1.
- Hàm mất mát Huber trên đích log(1 + Q) đã chuẩn hoá. E5 nhân thêm trọng số, E6 cộng thêm λ·phạt dự báo thiếu.
- Ở CV: dừng sớm theo tập xác thực (kiên nhẫn 20 epoch, tối đa 200). Ở huấn luyện cuối: số epoch cố định, không nhìn tập kiểm tra.
- Cấu hình đã chọn cho LSTM và GRU: L = 3, 128 nút ẩn, 2 lớp; E5: γ = 1, không chặn trên; E6: λ = 0,5.

**Đánh giá:**
- Chỉ tiêu: NSE, KGE, RMSE, MAE, PBIAS, NSE Q>Q95, Peak RMSE, Peak Bias, sai số thời điểm đỉnh.
- Mọi chỉ tiêu tính ở m³/s, chỉ trên ngày đích có quan trắc.
- So sánh cặp bằng bootstrap khối (khối 30 ngày, 2.000 lần) trong notebook 07 và 09.

**Thay đổi cấu hình** trong `experiments/config.yaml`:
- lưới siêu tham số: `models.<tên>.grid`;
- hạt giống: `seeds`;
- trọng số E5: `weights`;
- λ của E6: `penalty`;
- bootstrap: `bootstrap`.

Phần dữ liệu (`period`, `files`, `quality`, `imputation`, `horizons`, `splits`, `feature_groups`, `feature_sets`) nằm trong hash của manifest. Sửa phần này phải tạo phiên bản dữ liệu mới (mục 11).

---

## 8. Chạy bằng dòng lệnh (tuỳ chọn)

Mọi việc trên đều làm được bằng notebook. Dòng lệnh chỉ tiện khi muốn chạy nền hoặc chạy hàng loạt.

**Chạy các notebook theo thứ tự, ghi đầu ra vào chính các file notebook:**

```bash
python experiments/run_experiments.py                                   # 01 → 09, khoảng 22 phút
python experiments/run_experiments.py --only 05_LSTM.ipynb 07_SoSanh.ipynb
python experiments/run_experiments.py --only 10_ChayRieng_KichBan.ipynb
python experiments/run_experiments.py --fast                            # chạy thử nhanh, xem dưới
```

`--fast` là chế độ chạy thử máy, không dùng để lấy kết quả:
- dùng lưới nhỏ, 1–2 hạt giống, ít epoch; chạy đủ 9 notebook trong khoảng 2–3 phút;
- kết quả ghi vào `results_fast/`, notebook đã chạy ở `results_fast/_notebooks/`, dữ liệu đóng băng riêng ở `data/frozen/v1_fast/`;
- không đụng vào kết quả chính; chạy xong có thể xoá: `rm -rf results_fast data/frozen/v1_fast`.

Nếu không kích hoạt môi trường, thay `python` bằng `~/miniconda3/envs/MTL/bin/python`.

---

## 9. Kiểm thử

```bash
python -m pytest -q tests/
```

| File | Nội dung |
|---|---|
| `test_leakage.py` (24) | Không rò rỉ dữ liệu tương lai: điền thiếu nhân quả, chuẩn hoá theo train, cửa sổ, chia tập, đích |
| `test_data_consistency.py` (15) | .xlsx gốc ↔ .csv ↔ bảng ngày ↔ dữ liệu đóng băng; manifest; số mẫu |
| `test_final_models.py` (12) | Chỉ còn weight tốt nhất (E2, t+1); weight nạp lại dự báo đúng dự báo và chỉ tiêu đã lưu |
| `test_plan_scenario.py` (8) | Kịch bản KH: đúng ngày t+1, không thay đổi E2, sai số kế hoạch đúng thiết kế |

Notebook 02 cũng chạy pytest và lưu bản ghi vào `results/dataset/kiem_tra_toan_ven_pytest.txt`.

---

## 10. Cập nhật báo cáo sau khi chạy lại

Chỉ làm trên máy: `docs/` và `experiments/build_docs.py` không có trên GitHub.

```bash
python experiments/build_docs.py
```

Lệnh này làm ba việc:
- chép hình từ `results/` vào `docs/hinh/`;
- sinh lại Phụ lục B (toàn bộ bảng kết quả) và danh mục hình của báo cáo;
- sinh lại Phần B, C, D của đề cương.

Không sửa tay giữa các dòng `<!-- BEGIN/END -->`. Phần chữ của báo cáo (Phần 1–5) viết tay, nên phải cập nhật số liệu thủ công nếu kết quả thay đổi.

Nội dung các notebook được sinh từ `experiments/build_notebooks.py`. **Sinh lại một notebook sẽ xoá đầu ra đã lưu của nó**, nên chỉ dùng khi sửa nội dung notebook:

```bash
python experiments/build_notebooks.py --only 10
```

---

## 11. Nguyên tắc và lưu ý

- **Không sửa** `data/raw/` và `data/frozen/v1/`. Muốn thay đổi dữ liệu:
  1. đổi `dataset_version` trong `config.yaml`, ví dụ `v2`;
  2. chạy notebook 01, 02 để đóng băng phiên bản mới;
  3. chạy lại 03–09.

  Hàm `freeze()` từ chối ghi đè một phiên bản đã có nếu nội dung khác.
- **Không sửa** `src/data/*.py` khi vẫn dùng v1, vì hash mã nằm trong manifest. Kịch bản mới nên ghép biến ở tầng khác, như `src/plan_scenario.py`.
- **Mọi lựa chọn dựa trên kiểm định chéo.** Tập kiểm tra 2025–2026 chỉ để báo cáo, không dùng để chọn mô hình, kịch bản hay hạt giống.
- **Kịch bản mới:**
  1. viết thiết kế và tiêu chí đánh giá trước khi chạy (ví dụ `docs/ThietKe_KichBan_KeHoachXa.md`);
  2. báo cáo mọi kết quả, kể cả không thuận lợi;
  3. không loại hạt giống.
- **Tái lập:**
  - trên cùng GPU, kết quả trùng hoàn toàn: cuDNN tất định, hạt giống cố định;
  - notebook 10 (mục 3) đã được kiểm tra trùng kết quả chính ở mọi loại kịch bản (E2, E4, E5, E6, E56, FUT, PERS, KH, t+1 … t+10).

---

## 12. Xử lý sự cố

| Lỗi | Nguyên nhân, cách xử lý |
|---|---|
| VS Code không thấy kernel MTL | Cài extension Python và Jupyter; `Ctrl+Shift+P` → "Python: Select Interpreter" → chọn `~/miniconda3/envs/MTL/bin/python`; bấm lại Select Kernel. |
| `ModuleNotFoundError: No module named 'src'` | Kernel không phải MTL, hoặc notebook nằm ngoài thư mục dự án. Mở notebook ở thư mục gốc `BacSon/Ninh/`, chọn kernel MTL. |
| (Colab) `Chưa có dự án trên Google Drive` | Chưa chạy mục 0 của notebook 01: mở notebook 01, chạy ô 0a, 0b. |
| (Colab) `Không thấy …/T_DAThS_Ninh/Ninh_data.zip` | Tải `Ninh_data.zip` vào thư mục `T_DAThS_Ninh` trên Drive, giữ đúng tên; hoặc sửa `FILE_NEN` ở ô 0b. |
| (Colab) ô đầu in `Thiết bị: CPU` | Runtime → Change runtime type → T4 GPU, rồi chạy lại từ ô đầu. |
| (Colab) ô 0a in `Không cập nhật được (có file đã sửa trên Drive)` | Notebook trên Drive đã được lưu kèm đầu ra hoặc đã sửa. Muốn bỏ thay đổi và lấy bản mới: `!git -C /content/drive/MyDrive/T_DAThS_Ninh checkout -- .` rồi chạy lại ô 0a. |
| (Colab) lỗi khi nạp weight `.joblib` | Thư viện chưa đúng phiên bản: Runtime → Restart session, rồi chạy lại từ ô đầu (ô này cài `requirements_colab.txt` trước khi nạp thư viện). |
| `… đã bị thay đổi so với manifest` | Một file trong `data/frozen/v1/` bị sửa. Khôi phục bản gốc. |
| `cấu hình dữ liệu trong config.yaml khác với lúc đóng băng` | Đã sửa phần dữ liệu của `config.yaml`. Hoàn tác, hoặc tạo phiên bản dữ liệu mới. |
| `mã src/data đã thay đổi so với lúc đóng băng` | Đã sửa `src/data/*.py`. Hoàn tác, hoặc tạo phiên bản dữ liệu mới. |
| `chưa có cấu hình của … trong results/cau_hinh_chon.json` (notebook 10, mục 3) | Chưa chạy notebook của mô hình đó. Chạy notebook, hoặc đặt `TU_DAU = True`. |
| `không có weight đã lưu …` (notebook 10, mục 2) | Chỉ lưu weight tốt nhất (E2, t+1). Kịch bản khác: xem mục 1 hoặc chạy mục 3. |
| `torch.cuda.is_available()` = False | Cài lại torch đúng bản CUDA (mục 2), hoặc chạy trên CPU (chậm hơn, kết quả mạng hồi tiếp lệch rất nhỏ). |
