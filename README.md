# DSC 2026 — LegalQA Task 2 — Thành viên B (Retrieval & Chọn đoạn trích)

Đây là phần **B** trong hệ thống 4 mảnh của nhóm (A: chấm điểm/oracle eval, B: retrieval — thư mục này, C: generator, D/trưởng nhóm: điều phối chung).

> Thư mục này là 1 module trong monorepo — xem [`README.md` gốc](../README.md)
> để biết B khớp với `eval_harness/` (D) và `qa_generator/` (C) thế nào. Mọi
> lệnh dưới đây chạy với **CWD = thư mục này** (`retrieval/`), không phải gốc
> repo — `data/`, `data123/`, `data_retrieve/` nằm ở **`../`** (dùng chung với
> C/D, xem README gốc).

## Bắt đầu từ đâu

1. **`docs/SYSTEM_SCAFFOLD.md`** — bản đồ tổng, luôn cập nhật đúng hiện tại: cấu trúc thư mục, trạng thái từng module (B0-B7), thứ tự chạy pipeline, việc còn phải làm. **Đọc file này trước tiên.**
2. Docstring đầu mỗi file `.py` — spec hiện tại của riêng module đó (input/output/workflow).
3. `README.md` trong từng thư mục `src/bN_*/` — câu hỏi còn mở, lịch sử quyết định của module đó.

Nhật ký thực nghiệm chi tiết + báo cáo kỹ thuật đầy đủ được giữ riêng (tài liệu nội bộ, không
trong repo này) — hỏi trực tiếp nếu cần đào sâu hơn những gì `SYSTEM_SCAFFOLD.md` tóm tắt.

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

## 3 thư mục `data*` dễ nhầm

Cả 3 đều nằm ở **gốc repo** (`../data/` v.v. tính từ thư mục này), dùng chung
với `eval_harness/` và `qa_generator/` — code trong `retrieval/` tự resolve
đúng đường dẫn qua `src/common/config.py` (`config.DATA_DIR`,
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
**CWD = thư mục `retrieval/` này**:

| # | Bước | Script | Input cần có sẵn | Thời gian ước tính |
|---|---|---|---|---|
| 1 | Đặt dữ liệu gốc BTC vào `../data/` (gốc repo) | (thủ công — xin dữ liệu từ Trưởng nhóm/BTC, KHÔNG public) | — | — |
| 2 | Parse Điều/Khoản | `pipeline/build_parsed_corpus.py` | `data/corpus/` | vài phút |
| 3 | Build BM25 index | `pipeline/build_bm25_index.py` | `data/parsed_corpus.jsonl` | ~43 phút CPU |
| 4 | Nhúng corpus (Layer 2) | `kaggle_layer2/embed_corpus_notebook.py` (chạy trên **Kaggle GPU**, xem `kaggle_layer2/README.md`) | `data/parsed_corpus.jsonl` | ~2 giờ GPU |
| 5 | Sinh candidate cho tập cần chạy | `pipeline/build_layer3_candidates_public.py` (hoặc `_heldout.py`/`build_layer3_candidates.py` tuỳ tập) | Kết quả bước 3+4 | ~2 phút |
| 6 | Rerank Layer 3 | `kaggle_layer3/rerank_notebook.py` (chạy trên **Kaggle GPU**, xem `kaggle_layer3/README.md`) | Kết quả bước 5 | ~30 phút GPU |
| 7 | Đóng gói QA package cuối cùng | `pipeline/build_qa_packages_public_v1_top3.py` (hoặc bản khác) | Kết quả bước 6 | vài phút |

**Không cần làm lại bước 1-4 nếu chỉ muốn thử nghiệm nhanh trên tập nhỏ** — hỏi trong nhóm
xem có ai đã có sẵn `outputs/layer2/corpus_embeddings.npy` + `outputs/bm25_doc_index.pkl`
(2 file nặng nhất, tốn GPU/CPU nhất) để chia sẻ trực tiếp thay vì mỗi người tự train lại.
