# Hướng dẫn chạy Layer 2 (bi-encoder embedding) trên Kaggle GPU

## Vì sao cần Kaggle

Layer 2 cần tính embedding (vector ngữ nghĩa) cho **394,000 unit Khoản** bằng model
`BAAI/bge-m3` (~568M tham số, trong ngân sách ≤0.7B đã chốt). Chạy CPU sẽ rất chậm
(hàng giờ); GPU T4 free trên Kaggle làm việc này trong ~20-30 phút.

## Bước 1 — Tạo Kaggle Dataset (upload data)

**Đã chuẩn bị sẵn** — thư mục `kaggle_layer2/upload/` trong project đã có đủ 6 file, đúng
tên cần thiết (đổi tên sẵn để tránh trùng `train.json`/`warmup.json` giữa 2 nguồn dữ liệu):
`parsed_corpus.jsonl`, `data_retrieve_train.json`, `data_retrieve_warmup.json`,
`qa_train.json`, `qa_warmup.json`, `qa_public.json`.

Vào kaggle.com → "New Dataset" → kéo-thả **nguyên thư mục `kaggle_layer2/upload/`** (hoặc cả
6 file bên trong) vào. Đặt tên dataset — khuyên dùng đúng **`dsc-legalqa-b2-layer2-input`**
(khớp sẵn với `INPUT_DIR` trong code ở Bước 2, đỡ phải sửa gì) — nếu đặt tên khác thì nhớ
sửa lại biến `INPUT_DIR` ở CELL 2 của `embed_corpus_notebook.py`.

## Bước 2 — Tạo Notebook

1. kaggle.com → "New Notebook".
2. Bên phải, mục "Data" → "Add Input" → chọn dataset vừa tạo ở Bước 1.
3. Bên phải, mục "Settings" → **Accelerator: GPU T4 x2** (hoặc T4 x1 nếu không thấy x2) →
   **Internet: On** (bắt buộc, để tải model từ HuggingFace).
4. Copy toàn bộ nội dung `embed_corpus_notebook.py` (cùng thư mục với file này) vào các cell
   của notebook — mỗi đoạn có comment `# %% CELL n` là 1 cell riêng, copy đúng theo thứ tự.
5. Chạy lần lượt từng cell (Run All cũng được).

## Bước 3 — Tải kết quả về

**Có 2 cách — dùng CÁCH B nếu cách A tải không được (800MB+ dễ đứt giữa chừng qua trình
duyệt, đã gặp thật 13/08/2026):**

**Cách A (CELL 7)**: nén toàn bộ embedding thành `layer2_embeddings.zip` (~800MB-1GB) — tải
file này nếu mạng ổn định, dùng được cho cả việc SAU NÀY (wire Layer 2 vào production, cần
embedding thật ở máy local).

**Cách B (CELL 8+9, khuyên dùng nếu tải Cách A bị lỗi/đứt)**: tính bước SEARCH ngay trên
Kaggle (embedding đã có sẵn ở đó, không cần tải đi đâu), chỉ xuất ra **kết quả xếp hạng**
(danh sách văn bản đúng cho từng câu hỏi) — file `dense_runs.zip` chỉ vài MB, tải nhanh,
không lo đứt giữa chừng. Đủ để đo Recall@K/MRR so với BM25 ngay — chỉ khi nào cần WIRE
Layer 2 vào production thật mới cần quay lại tải Cách A.

## Bước 4 — Đưa lại cho tôi

- Nếu tải Cách A: giải nén `layer2_embeddings.zip` vào `outputs/layer2/`.
- Nếu tải Cách B: giải nén `dense_runs.zip` vào `outputs/dense_runs/`.

Báo tôi biết đã tải cách nào — tôi viết code đo Recall@K/MRR so với BM25 (đã có sẵn hạ tầng
`ir_metrics.py` từ trước) tương ứng.

## Xử lý lỗi thường gặp

**`FileNotFoundError` dù dataset đã "Add Input" đúng**: 2 nguyên nhân đã gặp thật (13/08/2026):
1. Dataset được add SAU KHI notebook đã chạy — phiên chưa mount kịp. Sửa: **Restart Session**
   (icon ⟳ cạnh "Draft Session" trên thanh công cụ), rồi chạy lại từ CELL 1.
2. Đường dẫn thật có thêm tiền tố `datasets/<username>/` (1 số tài khoản Kaggle bị vậy) —
   VD thật: `/kaggle/input/datasets/ngmaidnghi/dsc-legalqa-b2-layer2-input/` thay vì
   `/kaggle/input/dsc-legalqa-b2-layer2-input/`. Chạy `!find /kaggle/input/ -maxdepth 4` để
   tìm đường dẫn thật, rồi sửa `INPUT_DIR` ở CELL 2 cho khớp.

**`AttributeError: module 'sympy' has no attribute 'core'`** khi import `sentence_transformers`:
lỗi xung đột phiên bản thư viện có sẵn trong image Kaggle, không phải lỗi code/data. Sửa CELL
1 thêm `-U sympy` (đã có sẵn trong bản mới nhất của `embed_corpus_notebook.py`), sau đó BẮT
BUỘC **Restart Session** rồi chạy lại từ CELL 1 (không chỉ chạy lại đúng cell lỗi).

**Tải `layer2_embeddings.zip` (800MB+) bị đứt/không tải được**: dùng Cách B (CELL 8+9) thay
vì Cách A — xem Bước 3 bên dưới, xuất kết quả nhẹ thay vì tải nguyên embedding nặng.

## Thời gian ước tính

- Tải model (~1.2GB): 1-3 phút.
- Embed 394k Khoản: ~20-30 phút (GPU T4).
- Embed ~16.000 câu hỏi (data_retrieve train/warmup + QA train/warmup/public, làm luôn trong cùng phiên để đỡ
  phải quay lại Kaggle lần 2): ~3-5 phút.
- Tổng: dưới 40 phút, trong hạn mức GPU free hàng tuần của Kaggle (~30 giờ) rất thoải mái.
