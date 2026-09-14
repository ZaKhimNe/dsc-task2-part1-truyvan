# B3 — Đo Recall & chốt granularity

Trạng thái: **`eval_recall.py` đã code + chạy test sơ bộ trên data thật (06/08/2026)**. Còn 1 câu hỏi mở thật sự (xem bên dưới).

## Bài toán

Đo Recall@K của B2 theo từng phương án granularity (Điều nguyên khối vs Khoản + mở rộng ra trọn Điều khi trả kết quả), dùng nhãn từ B0 làm ground truth.

## Input / Output

Input: `labels` (output B0, `dict[question_id, list[MatchedSpan]]`) + `retrieved` (output B2, `dict[question_id, list[ScoredContext]]`, cùng bộ `question_id`, đã lấy sẵn top-K_max).
Output: `recall_at_k()` / `recall_at_multiple_k()` → `{"recall_at_k", "k", "n_evaluated", "n_no_label", "n_hits"}`. Câu hỏi B0 không gán được nhãn nào bị loại khỏi mẫu số (không tính là trượt), tỉ lệ này (`n_no_label`) là coverage B0, báo cáo riêng.

## Đã chạy — số sơ bộ (06/08/2026)

`experiments/demo_b3_step1_label.py` (B0, pool ngẫu nhiên 1,000/8,532 văn bản) → `experiments/demo_b3_step2_recall.py` (B2 trên 19 văn bản đúng + 300 nhiễu, n=21 câu hỏi có nhãn):

| K | Recall@K | hits/n |
|---|---|---|
| 5 | 0.905 | 19/21 |
| 10 | 0.905 | 19/21 |
| 20 | 0.905 | 19/21 |
| 50 | 1.000 | 21/21 |

**Không phải số chính thức** — pool nhỏ (319, không phải 8,532) → ít đối thủ cạnh tranh hơn thật, số nhiều khả năng lạc quan hơn; n=21 quá nhỏ để kết luận thống kê; granularity vẫn là nguyên văn bản (chưa chunk Khoản). Giá trị của bước này: xác nhận dây chuyền B0→B2→B3 chạy đúng bằng số thật, không lỗi logic.

## Câu hỏi mở còn treo — granularity chunk cuối cùng

Đề xuất đang treo trong handoff mục 5 — index ở Khoản (chính xác hơn cho B4), nhưng trả về trọn Điều chứa Khoản đó khi phục vụ (tránh hình phạt phân mảnh của METEOR, tối đa mất ước tính 0.5 điểm theo ghi chú của Trưởng nhóm — chưa tự verify). **Đây là chỉ số sở hữu #1 của cả hệ thống** — sai ở đây thì B4, B6, B7, và cả phần Generator của C đều vô nghĩa theo sau. Cần B1 (parser Điều/Khoản) chạy được trước khi so sánh 2 phương án granularity bằng Recall@K thật — B1 hiện `NotImplementedError`, chưa code thân hàm (không nằm trong scope ưu tiên hiện tại theo yêu cầu — chỉ làm phần retrieval).

## Giới hạn sandbox (không phải giới hạn thuật toán)

Cả B0 (`build_shingle_index`, đã thay thế) và B2 (`build_index`/underthesea) từng OOM trên sandbox 3.8GB RAM khi chạy full corpus/full pool trong giai đoạn đầu — 2 nguyên nhân khác nhau, đã sửa (máy hiện tại không còn giới hạn RAM này).
