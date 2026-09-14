# B4 — Chọn đoạn trích

Trạng thái: **đã code baseline v0** (`selection.py`, `select_span()`), test đối chiếu nhãn B0 đạt 67% (10/15). Còn 1 câu hỏi mở thật (xem bên dưới). Lưu ý: `select_span()` sau đó đo được có hại (xoá đáp án 55,6% số lần) và đã loại khỏi pipeline chính — xem cảnh báo trong docstring hàm.

## Bài toán

Từ 1 Khoản ứng viên (B1 chunk + B2 retrieve) và câu hỏi, chọn đơn vị con (Điểm/câu) liên quan nhất — khác retrieval, gần với evidence extraction. Ví dụ thật (`warmup.json` id 97213): Thông tư 64/2013/TT-BGTVT có 6 khoản, câu hỏi chỉ hỏi về đường bộ → chỉ Khoản 5 đúng, 5 khoản còn lại là nhiễu dù cùng document.

**Chỉ số sở hữu #2** — đo trực tiếp ~60% khối lượng điểm (phần trích nguyên văn, "Khối 2").

## Input / Output

Input: `question: str` + `khoan_text: str` (text 1 Khoản, từ `Dieu["khoan"][i]["text"]` sau B1).
Output: `str` — đoạn trích con trong `khoan_text`, nguyên văn.

## Thuật toán baseline v0 (đã code)

Tách `khoan_text` theo ranh giới Điểm tự nhiên (`\n\n`, B1 giữ lại sẵn) hoặc câu/mệnh đề nếu không chia Điểm → chấm điểm overlap token thô với câu hỏi → trả unit điểm cao nhất, fallback nguyên Khoản nếu không unit nào overlap. 0 tham số học, nhất quán với B2.

**Lý do KHÔNG dùng marker-based như B0** (`_extract_quote_core`): cách đó cần `answer` có sẵn để tách marker "như sau:"/"Theo đó" — B4 chạy serving-time chỉ có `question`, không có `answer`. Nhãn B0 sinh ra chỉ dùng để ĐÁNH GIÁ B4 (proxy thô, không phải input).

## Câu hỏi mở còn treo

Test hiện tại (67%) dùng nhãn B0 làm proxy — B0 định vị văn bản nguồn (document-level), không phải nhãn gold thật ở đúng granularity Điểm/câu mà B4 cần. Chưa thiết kế cách sinh nhãn gold riêng cho B4 ở granularity đúng — cần làm để có số đo chính thức thay vì proxy.

## Hướng nâng cấp đã nhận diện (chưa làm)

Case sai (`165287`, `82359`): chọn nhầm unit liền kề cùng chủ đề, khác điểm cụ thể — overlap token thô không đủ phân biệt khi nhiều Điểm dùng từ vựng giống nhau. Hướng khả thi: trọng số IDF, hoặc BM25 cấp câu thay vì overlap set thô.

## Tham khảo đáng đào tiếp

COLIEE Task 2 (case entailment) — chưa tra cứu chi tiết, làm khi có thời gian.
