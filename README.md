# DSC 2026 — LegalQA Task 2 — Truy vấn & Xếp hạng lại

Phần **truy hồi + chọn đoạn trích** (Thành viên B) của hệ thống LegalQA Task 2:
parse văn bản luật, hybrid retrieval (BM25 ∥ dense BGE-M3 → cross-encoder
rerank), đo recall, chọn Khoản, đóng gói schema cho Generator.

Repo này tách riêng từ monorepo của nhóm — chỉ chứa phần B. Phần sinh câu trả
lời (C) và bộ chấm điểm nội bộ (D) nằm ở repo khác.

> Mọi lệnh dưới đây chạy với **CWD = gốc repo này**. Dữ liệu BTC cấp nằm
> **ngoài repo**, ngang hàng với thư mục gốc (xem `.gitignore`): `../data/`,
> `../data123/`, `../data_retrieve/`.

## Bắt đầu từ đâu

1. **`docs/SYSTEM_SCAFFOLD.md`** — bản đồ tổng, luôn cập nhật đúng hiện tại: cấu trúc thư mục, trạng thái từng module (B0-B7), thứ tự chạy pipeline, việc còn phải làm. **Đọc file này trước tiên.**
2. Docstring đầu mỗi file `.py` — spec hiện tại của riêng module đó (input/output/workflow).
3. `README.md` trong từng thư mục `src/bN_*/` — câu hỏi còn mở, lịch sử quyết định của module đó.

4. **[`docs/BAO_CAO_THI_NGHIEM_RETRIEVAL.md`](docs/BAO_CAO_THI_NGHIEM_RETRIEVAL.md)**
   — báo cáo mọi thí nghiệm đã chạy trên truy vấn và reranker: ngưỡng đặt trước,
   kết quả, quyết định, **các hướng đã đóng (đừng thử lại)**, và cấu hình chốt.
   Đọc trước khi nghĩ hướng nâng cấp mới.

## Việc B làm (tóm tắt 1 câu mỗi bước)

```
B0  — có sẵn answer, tìm ngược xem trích từ Khoản nào trong corpus (tạo ground truth để CHẤM retrieval)
B1  — parse văn bản luật thành cấu trúc Điều/Khoản
B2  — retrieval: BM25 ∥ dense (BGE-M3) → union → Layer 3 rerank (bge-reranker-v2-m3)
B3  — đo recall/coverage của B2 (offline)
B4  — chọn top-3 Khoản cuối cùng để đưa cho Generator
B5  — quyết định phân bổ ngân sách tham số (đã chốt: 4 tỷ toàn hệ thống, B dùng 1,136/2,3 tỷ)
B6  — đóng gói kết quả B1+B2+B4 thành schema Generator (C) đọc được
B7  — KHÔNG PHẢI VIỆC CỦA B (thuộc A) — B chỉ cần output đúng schema cho B7 dùng
```

## Cấu hình chốt sau đợt thí nghiệm 19/09/2026

| Nút | Trước | Sau | Căn cứ |
|---|---|---|---|
| Nhúng cái gì | Khoản trần | **+ tiêu đề Điều** | E4, T1 |
| Xếp thứ tự | reranker toàn quyền | **RRF trọng số α = 0,7** | quét α + kiểm chéo hai nửa |
| Số đoạn giao | 3 | **2** | bảng §3.4 báo cáo |
| Ngân sách mỗi đoạn | 350 | **200** âm tiết | bảng §3.4 báo cáo |

```
s = α/(60 + dense_rank) + (1−α)/(60 + rerank_rank)
```

Build gói theo cấu hình chốt (CWD = gốc repo):

```
python pipeline/build_qa_packages_public_v8_tieu_de_khoan.py \
  --rerank-path outputs/layer3/L3_rerank_public_title.jsonl \
  --alpha 0.7 --max-contexts 2 --per-item 200 --expected-median 450
```

Chạy không tham số thì tái hiện cấu hình cũ. Tên file ra suy từ cấu hình để
không thể dán nhầm nhãn giữa hai bản build khác rổ. Lý do từng lựa chọn, kèm
các hướng **đã đóng**, nằm trong
[`docs/BAO_CAO_THI_NGHIEM_RETRIEVAL.md`](docs/BAO_CAO_THI_NGHIEM_RETRIEVAL.md).

## 3 thư mục `data*` dễ nhầm

Cả 3 đều nằm **ngoài repo**, ngang hàng với thư mục gốc (`../data/` v.v.) — code
tự resolve đúng đường dẫn qua `src/common/config.py` (`config.DATA_DIR`,
`config.DATA123_DIR`, `config.DATA_RETRIEVE_DIR`), không hardcode.

| Thư mục | Nội dung | Dùng để |
|---|---|---|
| `../data/` | Dữ liệu gốc BTC cấp — `train.json` (7.000 câu QA thật), `public-official.json` (1.000 câu ĐÍCH nộp bài), `corpus/` (8.532 văn bản luật) | Nguồn chính cho B0, B6, mọi đánh giá |
| `../data_retrieve/` | Nhãn context_id riêng của hạng mục Retrieval (7.500 câu — **tên file trùng nhưng nội dung khác** `data/train.json`) | Chỉ đo B2 cấp văn bản, KHÔNG phải data QA |
| `../data123/` | citation_metadata do C parse sẵn từ chính answer trong `data/train.json` | Build `outputs/citation_labels_*.json` — nhãn Khoản đối chứng độc lập với B0 |

## Trạng thái hiện tại (03/09/2026)

- ✅ **Baseline B0→B6 hoàn chỉnh, đã chạy end-to-end trên `data/public-official.json`** (1.000 câu tập ĐÍCH nộp bài) — không còn là việc "chưa làm", đã có **3 bản nộp thật lên leaderboard** (`outputs/qa_packages_public_v1_top3.json` = 0,4506 điểm, tốt nhất hiện tại; `v2_top1` = 0,4430; `v3_top1_raw` đang chờ điểm).
- Layer 3 rerank đã CHỐT GIỮ (xác nhận qua 2 nguồn nhãn độc lập, +10-14đ% Hit@1 cấp Khoản trên tập giữ kín).
- **Trước khi tự nghĩ hướng nâng cấp mới**: hỏi trực tiếp về các hướng đã thử và đóng lại (RRF, `max_per_doc=1`, `select_span`, ràng buộc đa dạng văn bản...) — đều đã được đo và **đóng lại bằng số liệu cụ thể**, đừng mất công đo lại đúng những thứ đó.
- Chi tiết đầy đủ từng module: xem bảng "Trạng thái từng module" trong `docs/SYSTEM_SCAFFOLD.md`.

## Cài đặt

```
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet')"
```

## Tái tạo dữ liệu — BẮT BUỘC đọc trước khi chạy bất kỳ script nào

Repo này **không chứa dữ liệu** (BTC cấp + mọi file trung gian tự sinh, xem `.gitignore`) —
tổng cộng ~2,5GB, không hợp để nằm trong git. Sau khi clone, cần tự tạo lại theo ĐÚNG THỨ TỰ
sau (mỗi bước phụ thuộc bước trước, không nhảy cóc). Chạy tất cả lệnh dưới đây với
**CWD = gốc repo này**:

| # | Bước | Script | Input cần có sẵn | Thời gian ước tính |
|---|---|---|---|---|
| 1 | Đặt dữ liệu gốc BTC vào `../data/` (ngoài repo) | (thủ công — xin dữ liệu từ Trưởng nhóm/BTC, KHÔNG public) | — | — |
| 2 | Parse Điều/Khoản | `pipeline/build_parsed_corpus.py` | `data/corpus/` | vài phút |
| 3 | Build BM25 index | `pipeline/build_bm25_index.py` | `data/parsed_corpus.jsonl` | ~43 phút CPU |
| 4 | Nhúng corpus (Layer 2) | `kaggle_layer2/embed_corpus_notebook.py` (chạy trên **Kaggle GPU**, xem `kaggle_layer2/README.md`) | `data/parsed_corpus.jsonl` | ~2 giờ GPU |
| 5 | Sinh candidate cho tập cần chạy | `pipeline/build_layer3_candidates_public.py` (hoặc `_heldout.py`/`build_layer3_candidates.py` tuỳ tập) | Kết quả bước 3+4 | ~2 phút |
| 6 | Rerank Layer 3 | `kaggle_layer3/rerank_notebook.py` (chạy trên **Kaggle GPU**, xem `kaggle_layer3/README.md`) | Kết quả bước 5 | ~30 phút GPU |
| 7 | Đóng gói QA package cuối cùng | `pipeline/build_qa_packages_public_v1_top3.py` (hoặc bản khác) | Kết quả bước 6 | vài phút |

**Không cần làm lại bước 1-4 nếu chỉ muốn thử nghiệm nhanh trên tập nhỏ** — hỏi trong nhóm
xem có ai đã có sẵn `outputs/layer2/corpus_embeddings.npy` + `outputs/bm25_doc_index.pkl`
(2 file nặng nhất, tốn GPU/CPU nhất) để chia sẻ trực tiếp thay vì mỗi người tự train lại.
