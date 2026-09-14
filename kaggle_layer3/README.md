# Hướng dẫn chạy Layer 3 (reranker) trên Kaggle GPU

## Vì sao cần Kaggle (khác Layer 2)

Layer 2 (bi-encoder) mã hoá câu hỏi/văn bản RIÊNG BIỆT — tính embedding 1 lần, search sau đó
rất rẻ (CPU đủ nhanh). Layer 3 (reranker, cross-encoder) đọc CẢ câu hỏi VÀ candidate CÙNG LÚC
qua model cho MỖI cặp — không tính trước được. Đo thật trên CPU máy local (13/08/2026):
**954ms/candidate** (candidate thật dài 134-2.241 ký tự, không phải text ngắn giả lập) —
ngoại suy 7.000 câu × 30 candidate sẽ mất **hàng chục giờ trên CPU**. GPU T4 cần thiết.

## Khác biệt với Layer 2: KHÔNG cần upload lại corpus 700MB

Phần tìm candidate (`search_units_hybrid` — BM25 ∥ dense, union cấp unit) không cần GPU,
đã chạy sẵn ở máy local (`pipeline/build_layer3_candidates.py`) — chỉ cần upload **1 file
chứa top-50 candidate/câu hỏi đã chọn sẵn** (`layer3_candidates_train.jsonl`, ~368MB — nhẹ
hơn nhiều so với toàn bộ `parsed_corpus.jsonl` (~700MB) vì chỉ chứa candidate đã lọc, không
phải cả kho văn bản).

Bản 17/08 (chốt kiến trúc cuối) — 7.000 câu, sinh trong 810.6s, 0 candidate rỗng (đã kiểm
tra). KHÁC bản trước (top-30, chỉ BM25+rerank BM25): giờ dùng `search_units_hybrid` (union
BM25 + dense, top-50, không ràng buộc đa dạng).

## Bước 1 — Tạo Kaggle Dataset

File cần upload: `kaggle_layer3/upload/layer3_candidates_train.jsonl` (sinh ra bởi
`pipeline/build_layer3_candidates.py`, chạy ở máy local trước — xem log
`build_layer3_candidates.log` để biết đã xong chưa và kích thước file).

Vào kaggle.com → "New Dataset" → upload file này. Đặt tên dataset — khuyên dùng đúng
**`dsc-legalqa-b2-layer3-input`** (khớp sẵn `INPUT_DIR` trong code).

## Bước 2 — Tạo Notebook

1. kaggle.com → "New Notebook".
2. Panel phải → "Add Input" → chọn dataset vừa tạo.
3. Panel phải → Settings → **Accelerator: GPU T4 x2** → **Internet: On**.
4. Copy từng đoạn `# %% CELL n` trong `rerank_notebook.py` (cùng thư mục) vào notebook, đúng
   thứ tự.
5. Run All.

## Bước 3 — Tải kết quả về

File `layer3_results.zip` — RẤT NHẸ (chỉ vài MB, không phải embedding nặng như Layer 2).
Giải nén vào `outputs/layer3/` trong project, báo tôi.

Lưu ý: output giữ nguyên **cấp unit** (`ranked_units`, mỗi phần tử có `unit_id`, `context_id`,
`score`) — KHÔNG collapse về văn bản. Lý do: bài học từ B4 (17/08) — đo/xếp hạng ở cấp văn
bản đánh lừa, cấp Khoản mới là cái quyết định Hit thật. Việc đánh giá Layer 3 có thật sự cải
thiện hay không phải làm ở cấp Khoản bằng nhãn B0, giống cách đã làm với `max_per_doc`.

## Lưu ý — các lỗi đã gặp ở Layer 2, có thể lặp lại

Xem `kaggle_layer2/README.md` mục "Xử lý lỗi thường gặp" — 3 lỗi đã gặp thật (dataset chưa
mount kịp cần Restart Session, đường dẫn `datasets/<username>/`, xung đột `sympy`) đều có
thể lặp lại ở đây, cách sửa giống hệt.

## Thời gian ước tính

Chưa đo được tốc độ GPU thật (đang chuẩn bị) — sẽ cập nhật sau khi có số đo đầu tiên.
