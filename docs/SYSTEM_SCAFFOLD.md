# Khung hệ thống — Thành viên B (Retrieval & Chọn đoạn trích)

> Bản đồ tổng cho `src/`. File này chỉ nối các mảnh lại và cho biết bắt đầu từ đâu.
>
> **Nguồn spec chính thức**: file phân công gốc "DSC2026 — Phân công nhóm Task 2: LegalQA" (13 trang, Trưởng nhóm, 02/08/2026, trong base knowledge). Bám sát tài liệu này khi hợp lý, nhưng KHÔNG khoá cứng cách giải quyết theo đó nếu thực nghiệm cho thấy hướng khác tốt hơn.

## Quy ước tài liệu (3 vai trò, không trùng lặp)

| File | Vai trò | Khi nào đọc |
|---|---|---|
| Docstring đầu `*.py` | Spec HIỆN TẠI: bài toán, input/output, workflow. | Mỗi lần mở file để code. |
| `README.md` mỗi module | Câu hỏi CÒN MỞ, rationale, tham khảo. | Khi cần ra quyết định hoặc review lại. |
| File này (`SYSTEM_SCAFFOLD.md`) | Bản đồ LUÔN PHẢI ĐÚNG với hiện tại — duy nhất trong repo này. | Khi cần biết "hiện tại đang ở đâu, file nào ở đâu". |

Nhật ký thực nghiệm chi tiết + báo cáo kỹ thuật đầy đủ (10 quyết định kèm số liệu, phân tích
giới hạn...) được giữ riêng dạng tài liệu nội bộ (PDF), không nằm trong repo này — liên hệ trực
tiếp nếu cần đào sâu 1 quyết định cụ thể hơn những gì file này tóm tắt.

## Cấu trúc thư mục (cập nhật 13/09/2026 — sau khi gộp monorepo với C/D)

Thư mục này (`retrieval/`) là 1 module trong monorepo — nằm ngang hàng với
`eval_harness/` (D) và `qa_generator/` (C) ở gốc repo, xem `../README.md`. Cây
dưới đây chỉ vẽ bên trong `retrieval/`; `data/`, `data_retrieve/`, `data123/`
nằm ở **`../`** (gốc repo, dùng chung với C/D — không thuộc riêng `retrieval/`
nữa), tự resolve qua `config.DATA_DIR`/`config.DATA123_DIR`/
`config.DATA_RETRIEVE_DIR` (`src/common/config.py`, anchor bằng `REPO_ROOT`).

```
retrieval/                     (thư mục này — B0-B7, xem ../README.md cho C/D)
├── kaggle_layer2/                hạ tầng ĐÃ DÙNG XONG cho Layer 2 (bi-encoder) trên Kaggle GPU —
│                                 README.md + embed_corpus_notebook.py (thư mục upload/ đã xoá, tái
│                                 tạo lại được từ data/ nếu cần chạy lại)
├── kaggle_layer3/                hạ tầng Layer 3 (reranker) trên Kaggle GPU — README.md +
│                                 rerank_notebook.py, upload/layer3_candidates_heldout.jsonl (candidate
│                                 800 câu heldout — bản train 7.000 câu đã xoá 31/08, tái tạo được bằng
│                                 pipeline/build_layer3_candidates.py nếu cần chạy lại)
├── docs/                        SYSTEM_SCAFFOLD.md (file này) — nhật ký/báo cáo chi tiết giữ
│                                 riêng dạng tài liệu nội bộ, không trong repo
├── src/
│   ├── common/                   config.py (CORPUS_DIR trỏ data/corpus/), io_utils.py, scoring.py
│   ├── b0_autolabel/             ✅ dùng B2 search_units() tìm candidate, SW align cấp Khoản
│   ├── b1_parser/                ✅ parse_document() — dieu_id/khoan_id/char_start/char_end/parse_status/phu_luc_raw
│   ├── b2_retrieval/             ✅ THIẾT KẾ CUỐI: search_units_hybrid() = Layer 1 (BM25 doc) ∥ Layer 2
│   │                              (dense BGE-M3, unit-level) → Union cấp unit → xếp hạng dense → top-50
│   │                              → Layer 3 (reranker, ĐÃ CHỐT GIỮ — cải thiện xác nhận cả cấp văn bản
│   │                              +5,0đ% và cấp Khoản +10-14đ% Hit@1 trên tập giữ kín, 2 nguồn nhãn độc
│   │                              lập B0+citation, 31/08) — vẫn là bước BATCH riêng qua Kaggle, không
│   │                              gọi trực tiếp trong hàm chính vì không cần suy luận sống
│   ├── b3_eval_recall/           ✅ eval_recall.py (B0-based) + ir_metrics.py (tổng quát, dùng cho data_retrieve)
│   ├── b4_span_selection/        ✅ `select_top_n_khoan` (top-3, ĐANG DÙNG) — `select_span` ĐÃ LOẠI KHỎI
│   │                              pipeline chính (đo có hại, xoá đáp án 55,6% số lần), giữ code + docstring
│   │                              cảnh báo, không xoá hẳn
│   ├── b5_architecture_tradeoff/ ✅ ĐÃ CHỐT — giữ cả BM25+dense (bổ trợ nhau). Ngân sách TOÀN HỆ THỐNG
│   │                              4 tỷ (xác nhận qua C 31/08 — SỬA từ số cũ 2,2 tỷ đã lỗi thời), C dùng
│   │                              1,7 tỷ, B được cấp 2,3 tỷ, đang dùng 1,136 tỷ (dư 1,164 tỷ)
│   └── b6_context_package/       ✅ QAPackage (id/question/contexts/reference_answer). ContextItem =
│                                  document/document_number/article/clause/text/retrieval_score (schema
│                                  xác nhận với C 31/08 — article+clause tách rời, thêm document_number).
│                                  `write_qa_packages_json()` ghi thẳng 1 file `.json` (không còn bước
│                                  `.jsonl` trung gian — đơn giản hoá, chỉ 1.000-2.000 record/lần chạy nên
│                                  không cần streaming). Còn treo: cấu trúc field `document` (Generator
│                                  chưa xác nhận). (Đã bỏ thư mục `b7_oracle_eval/` — không phải việc của
│                                  B, thuộc A)
├── pipeline/                    build/eval script PHẢI CHẠY ĐÚNG THỨ TỰ khi cần tái tạo — xem "Thứ tự
│                                 chạy" bên dưới. Gồm cả script đo lường (measure_*, eval_*, crosscheck_*,
│                                 sample_*) không sinh output cho bước sau, chỉ để lấy số liệu.
├── experiments/                 demo/test throwaway (demo_b0/b1/b2/b4_*, demo_b_end_to_end.py)
├── outputs/
│   ├── runs/                     run file BM25 + dense (L1_doc, L1_khoan, Layer15, L2_dense, gold_*) —
│   │                             bằng chứng thô cho các bảng ablation trong báo cáo, không xoá
│   ├── layer2/                   embedding Layer 2 từ Kaggle (corpus + 5 bộ query embedding, ~845MB —
│   │                             đã xoá bản `.before_patch.npy` backup 31/08, patch đã xác nhận ổn định)
│   ├── layer3/                   kết quả rerank Layer 3 từ Kaggle — L3_rerank_train.jsonl (7.000 câu
│   │                             data_retrieve, cấp văn bản) + L3_rerank_heldout.jsonl (800 câu giữ
│   │                             kín, cấp Khoản) — CẢ HAI đã đo, đã CHỐT giữ Layer 3
│   ├── bm25_doc_index.pkl        Layer 1 (index chính, đang dùng)
│   ├── bm25_khoan_index.pkl      Layer 1.5 bản BM25 (dự phòng, không phải mặc định — xem b2_retrieval)
│   ├── doc_number_index.json     index số hiệu văn bản → context_id (build_doc_number_index.py, 31/08),
│   │                             dùng cho citation_labels + document_number trong QAPackage
│   ├── citation_labels_{train,validation,test}.json  nhãn Khoản/Điều từ data123 citation_metadata
│   │                             (build_citation_labels.py, 31/08) — nguồn đối chứng B0, ĐỘC LẬP với B0
│   ├── b0_labels_warmup.json     nhãn B0 trên warmup.json (500 câu)
│   ├── b0_labels_train_sample.json  nhãn B0 dev-sample 1.500 câu (ĐÃ DÙNG ĐỂ TUNE — không dùng lại
│   │                             để ra quyết định thiết kế mới)
│   ├── b0_labels_train_heldout.json  nhãn B0 TẬP GIỮ KÍN 800 câu (KHÔNG DÙNG ĐỂ TUNE — chỉ đánh giá,
│   │                             chỉ 198/800 = 24,8% đạt confidence≥0.6 — coverage thấp, xem eval)
│   ├── qa_packages_heldout.json  800 QAPackage (schema Generator) cho 800 câu giữ kín — DÙNG ĐỂ GỬI
│   │                             CHO C chạy oracle eval nội bộ, KHÔNG PHẢI bài nộp final
│   └── LAYER3_EVALUATION_SUMMARY.md  báo cáo đánh giá Layer 3 trên 2 nguồn nhãn (B0 + citation), 31/08
└── requirements.txt
```

**Lưu ý đã dọn dẹp 31/08 (lần 2)**: xoá `outputs/layer2/corpus_embeddings.before_patch.npy` (885MB,
backup trước patch bug B1 — patch đã xác nhận ổn định), `kaggle_layer3/upload/layer3_candidates_train.jsonl`
(386MB, đã upload Kaggle xong, tái tạo được), `layer3_results/` (28MB, trùng `outputs/layer3/`),
`outputs/qa_packages_heldout.jsonl` (5,5MB, đã gộp vào `.json`) — tổng ~1,3GB giải phóng.

**Lưu ý đã dọn dẹp 20/08 (lần 1)**: xoá 21 file log/txt rác ở gốc thư mục, `data_retrieve/selected-contexts/` (485MB, trùng lặp không dùng), `kaggle_layer2/upload/` (689MB, đã dùng xong), 2 index BM25 của config ablation đã loại (`_name`, `_nocap`, 168MB) — tổng ~1,34GB giải phóng.

## Luồng dữ liệu — 5 nguồn, mỗi nguồn chia khác nhau

### 1. `data/train.json` (7.000 câu QA thật) — chia 3 phần KHÔNG TRÙNG NHAU

| Phần | Số câu | Tỉ lệ | Cách chọn | Vai trò |
|---|---|---|---|---|
| Dev-sample | 1.500 | 21,4% | `random.sample(seed=42)` (`build_b0_labels_train_sample.py`) | **ĐÃ DÙNG TUNE** — chốt `max_per_doc`, bỏ `select_span`, so RRF (17/08). KHÔNG dùng lại để ra quyết định mới. |
| Heldout | 800 | 11,4% | Pool = 7.000 − 1.500 dev-sample, `random.Random(seed=123).sample()` (`build_b0_labels_train_heldout.py`) | **TUYỆT ĐỐI KHÔNG TUNE** — chỉ đánh giá Layer 3 + context expansion 1 lần (30-31/08). |
| Chưa đụng | 4.700 | 67,1% | Phần còn lại | Chưa từng gán nhãn B0/dùng trong nghiên cứu retrieval. |

2 seed khác nhau (42 vs 123) có chủ đích — tránh 2 tập trùng lặp dù cùng cơ chế random.

### 2. `data/warmup.json` (500 câu) — không chia, label toàn bộ

Dùng sớm nhất (06/08, giai đoạn code B0 lần đầu) — `build_b0_labels.py` gán nhãn full 500 câu (313/500 = 62,6% có nhãn). File riêng, không thuộc luồng tune/heldout của `train.json`.

### 3. `data/public-official.json` (1.000 câu) — không chia, chạy full pipeline

`answer = null`, chạy trọn B0→B6 → 3 bản nộp `qa_packages_public_v1/v2/v3` lên leaderboard (03/09).

### 4. `data_retrieve/` (7.500 câu = train 7.000 + warmup 500, KHÁC NỘI DUNG `data/`) — không chia tune/test

Nhãn `context_id` có sẵn từ nguồn ngoài (không tự sinh) nên không sợ kiểu "tune" như B0 tự nhãn — nhưng vẫn bị dùng LẶP LẠI nhiều lần cho nhiều quyết định liên tiếp (ablation Layer 1, RRF, `max_per_doc`) → log 17/08 tự cảnh báo đây là dạng data snooping nhẹ, số tuyệt đối có thể lạc quan hơn thực tế.

### 5. `data123/` (7.000 câu, cùng nội dung answer với `data/train.json`) — C chia theo 80/10/10

| Split | Số câu | Tỉ lệ |
|---|---|---|
| train | 5.600 | 80% |
| validation | 700 | 10% |
| test | 700 | 10% |

**2 con số KHÁC GIAI ĐOẠN, cả 2 đều đúng — đã xác minh lại bằng dữ liệu thật (verify script chạy trực tiếp trên `outputs/citation_labels_*.json`, không phải suy đoán):**

| Con số | Ý nghĩa | Giai đoạn |
|---|---|---|
| **2.946/7.000 = 42,1%** | `baseline_eligible` — C đánh dấu "đủ rõ để thử" | Đầu vào, TRƯỚC khi B lọc |
| **1.375/7.000 = 19,6%** | Labeled THÀNH CÔNG (khoan+dieu) sau `build_citation_labels.py` | Đầu ra, SAU khi B lọc thêm (chỉ nhận đúng 1 văn bản + 1 Điều + tối đa 1 Khoản, map được số hiệu, unit tồn tại thật trong `parsed_corpus`) |

Trong tập heldout 800 câu: **78 câu `unit_type="khoan"`** (dùng trong mọi báo cáo Layer 3/error-analysis) + **78 câu `unit_type="dieu"`** (khớp đúng Điều nhưng không xác định được Khoản — HIỆN CHƯA DÙNG trong bất kỳ phép đo nào, dữ liệu có sẵn nhưng đang bỏ phí).

## Thứ tự chạy pipeline (build script, khi cần tái tạo từ đầu)

Bước build index cấp Khoản (`outputs/bm25_khoan_index.pkl`, dùng cho Layer 1.5 dự phòng) nằm
trong 1 script ablation lịch sử giữ riêng ngoài repo — nếu cần, tự viết lại bằng
`retrieve.build_index()` trên dict `{khoan_id: text}` duyệt từ `parsed_corpus.jsonl` (xem
`expand_to_units()` trong `src/b2_retrieval/retrieve.py` để biết cách duyệt đúng cấu trúc).

```
build_parsed_corpus.py → build_bm25_index.py (Layer 1, cache) → build_b0_labels.py (B0 warmup)
    → build_b0_labels_train_sample.py (B0 dev-sample 1.500 câu, ĐÃ DÙNG TUNE)
    → build_b0_labels_train_heldout.py (B0 TẬP GIỮ KÍN 800 câu, chạy theo lô resumable — KHÔNG dùng tune)
    → [Kaggle] embed_corpus_notebook.py → patch_layer2_embeddings.py (vá embedding sau khi sửa bug B1)
    → build_layer3_candidates.py / build_layer3_candidates_heldout.py → [Kaggle] rerank_notebook.py
    → eval_layer3_rerank.py / eval_layer3_rerank_heldout.py (đo đóng góp Layer 3)
    → build_doc_number_index.py → build_citation_labels.py (nhãn Khoản đối chứng B0, từ data123)
    → eval_layer3_citation_labels_heldout.py (đối chiếu Layer 3 trên 2 nguồn nhãn độc lập)
    → error_analysis_heldout.py (phân loại Loại A/B case Layer 3 miss)
    → eval_context_expansion_heldout.py (quyết định TOP_N + có mở rộng cả Điều không)
    → build_qa_packages_heldout.py (đóng gói 800 câu giữ kín theo schema Generator, gửi C — TOP_N=1 +
    mở rộng Điều, quyết định 31/08)
    → measure_chunking_ceiling.py, crosscheck_b0_labels.py, sample_b0_audit.py (đo lường độc lập, không
    sinh output cho bước sau)
```

**ĐÃ CHẠY (03/09/2026)**: sinh candidate + Layer 3 + qa_packages cho `data/public-official.json` (1.000 câu, tập ĐÍCH nộp bài) — 3 bản nộp (`qa_packages_public_v1_top3.json` = 0,4506 điểm LB, tốt nhất; `v2_top1` = 0,4430; `v3_top1_raw` chờ điểm). Baseline coi như đóng.

## Trạng thái từng module (cập nhật 31/08/2026)

| Module | Trạng thái | Việc còn lại |
|---|---|---|
| B0 autolabel | ✅ ổn định, độ tin đã kiểm chứng khách quan (61,9%) — nhưng COVERAGE thấp (198/800 = 24,8% tập giữ kín đạt confidence≥0.6) | Cân nhắc hạ `B0_CONFIDENCE_TRUST` (có thể đang loại nhãn đúng không cần thiết) |
| B1 parser | ✅ full corpus, 3/4 bug đã sửa (kể cả bug text rỗng 7.163 unit, 17/08) | Bug danh sách con đánh số trùng (7.632 `khoan_id` trùng) — ghi nhận, chưa sửa |
| B2 retrieval | ✅ `search_units_hybrid()` là thiết kế cuối, đã wire đủ Layer 1+1.5+2+Union; Layer 3 ĐÃ CHỐT GIỮ (xác nhận cả 2 cấp, 2 nguồn nhãn độc lập B0+citation) | Formalize Layer 3 thành quy trình batch chuẩn cho `public-official.json` (CHƯA CHẠY — xem trên) |
| B3 eval recall | ✅ đầy đủ, kể cả đo trần chunking (31,1% câu cần thông tin ngoài 1 Khoản) | ĐÃ QUYẾT 31/08: mở rộng context sang cả Điều (không phải Khoản lân cận) — xem dưới |
| B4 span selection | ✅ `select_top_n_khoan` — TOP_N ĐỔI 3→1 (31/08, xem B6) | — |
| B5 | ✅ ĐÃ CHỐT, ngân sách 4 tỷ toàn hệ thống đã xác nhận chính thức qua C (31/08) | — |
| B6 context package | ✅ schema xác nhận với C 31/08 (article/clause tách rời + document_number). **ĐẢO NGƯỢC quyết định top-3**: C xác nhận Generator copy máy móc, chỉ xử lý 1 context/câu — `qa_packages_heldout.json` giờ TOP_N=1 + context mở rộng cả Điều (Hit rate 46,5%→52,2%, xem `eval_context_expansion_heldout.py`) | Xác nhận field `document` với Generator (còn treo) |
| B7 oracle eval | ❌ KHÔNG PHẢI VIỆC CỦA B (thuộc A) | B chỉ cần đảm bảo `qa_packages_*.json` đúng schema, gửi C/A dùng |

**Bảng số liệu B2 chốt cuối cùng — Layer 3, 2 nguồn độc lập** (xem `outputs/LAYER3_EVALUATION_SUMMARY.md`, tự sinh lại bằng `pipeline/eval_layer3_rerank.py`/`eval_layer3_rerank_heldout.py`):

| K | Cấp văn bản (n=7.000, `data_retrieve`) | Cấp Khoản, B0 gold (n=198) | Cấp Khoản, citation gold (n=78) |
|---|---|---|---|
| 1 | — | 35,9% → **46,0%** (+10,1đ%) | 33,3% → **47,4%** (+14,1đ%) |
| 3 | 64,9% → **69,9%** (+5,0đ%) | 54,0% → **59,6%** (+5,6đ%) | 56,4% → **62,8%** (+6,4đ%) |
| 50 | 87,3% → 87,3% (không đổi) | 86,4% → 86,4% (không đổi) | 80,8% → 80,8% (không đổi) |

3 nguồn độc lập cho cùng xu hướng, "không đổi" tại K=50 xác nhận phép đo đúng lý thuyết — **CHỐT giữ Layer 3**. Lưu ý: citation gold chỉ phủ 78/800 (9,8%) vì `data123` chỉ parse chắc chắn được 19,6% của 7.000 câu — xem "Việc nên làm tiếp theo" nếu muốn mở rộng lên 506 câu (toàn bộ citation "fresh chưa tune").

**Error analysis Layer 3 (31/08, `error_analysis_heldout.py`, n=226 union gold)**: Hit@10 sau rerank 76,1% (172/226). 54 case thất bại: **68,5% Loại A** (retrieval miss — gold chưa từng vào top-50 trước Layer 3, lỗi B1/B2) vs **31,5% Loại B** (reranker miss — gold có trong top-50 nhưng bị đẩy ngoài top-10 sau). Trong Loại B, **76,5% (13/17) là Layer 3 làm TỆ HƠN dense gốc**, gồm 4 case dense đã xếp top-6 (2 case ở rank 1) bị đẩy hẳn ra ngoài top-10 — rủi ro cụ thể cần lưu ý dù kết luận tổng thể "Layer 3 tốt" vẫn đúng.

**Context expansion (31/08, `eval_context_expansion_heldout.py`)**: đo Hit rate (gold nằm trong text context đã trả) so (a) chỉ Khoản vs (b) cả Điều, trên n=226:
| TOP_N | (a) chỉ Khoản | (b) cả Điều | Chênh lệch | Độ dài TB (a→b) |
|---|---|---|---|---|
| 3 (đã bỏ, xem dưới) | 60,2% | 66,4% | +6,2đ% *** | 2.241→7.015 ký tự |
| 1 (ĐANG DÙNG) | 46,5% | 52,2% | +5,8đ% *** | 734→2.322 ký tự |

**QUYẾT ĐỊNH ĐÃ ÁP DỤNG (31/08)**: C xác nhận Generator copy máy móc đoạn trích luật (không LLM tự chọn giữa nhiều context) → chỉ xử lý ĐÚNG 1 context/câu hỏi, đảo ngược hẳn quyết định top-3 (13/08). `build_qa_packages_heldout.py` đã sửa: `TOP_N=1` + context mở rộng trả cả Điều (bù lại việc mất top-2/3 dự phòng). `qa_packages_heldout.json` đã build lại theo schema mới — verify: 800/800 record đúng 1 context.

## Việc nên làm tiếp theo

**Baseline B0→B6 đã đóng** (chạy xong end-to-end trên `public-official.json`, có điểm LB thật).
Trước khi tự nghĩ hướng nâng cấp mới — **hỏi trực tiếp** về các hướng đã thử và đóng lại
(RRF làm sụp Hit@1, `max_per_doc=1` "thắng giả" ở cấp văn bản nhưng sập Hit cấp Khoản,
`select_span` xoá mất đáp án 55,6% số lần) — tránh mất công đo lại đúng những thứ đã có
kết luận rõ ràng bằng số liệu.

## Danh sách câu hỏi/việc mở cần xác nhận

- **Generator (C)**: cấu trúc field `document` trong `ContextItem` (B6) — đang dùng tạm `{"name","link"}`, Generator ghi "metadata hoặc parse từ corpus" (mơ hồ). Các field khác đã xác nhận 31/08.
- **Trưởng nhóm**: tập dev nội bộ 150 câu (T2) đã chốt danh sách chưa? Harness chấm điểm chung đã có chưa?
