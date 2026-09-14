# B6 — Context package (đóng gói context cho Generator/C)

Trạng thái: **đã code, đang dùng** (`package.py`). Xem docstring đầu file để biết schema hiện tại — README này chỉ giữ câu hỏi mở/lịch sử quyết định, KHÔNG lặp lại schema (dễ lệch nhau khi 1 bên quên cập nhật).

## Bài toán

Đóng gói context (kết quả từ B1+B2+B4) thành format Generator (C) tiêu thụ được. **Điểm dừng của B** — không sinh câu trả lời cuối.

## Lịch sử schema (mới nhất trước)

- **(sau 03/09/2026)** — đơn giản hoá: bỏ hẳn bước ghi `.jsonl` trung gian rồi gộp — `jsonl_to_json()` đổi thành `write_qa_packages_json()`, ghi thẳng 1 file `.json` từ list package trong RAM. Lý do bỏ streaming: mỗi lần chạy chỉ 800-1.000 record, không cần an toàn "ghi từng dòng khi bị ngắt giữa chừng" như `parsed_corpus.jsonl` (8.512 record, chạy vài phút).
- **31/08/2026** — xác nhận với C: `article`/`clause` tách rời (không gộp chuỗi), thêm `document_number`.
- **13/08/2026** — rebuild theo schema Generator đề xuất: 1 record = 1 CÂU HỎI (`contexts` là list, không phải 1 record/1 context như bản đầu), thêm `question` + `reference_answer` vào package (trước đó B6 không chạm answer).
- **06/08/2026 (bản đầu, đã bỏ)** — `ContextPackage` phẳng: `question_id, context_id, dieu_so, khoan_so, span_text, source_name, source_link, retrieval_score`, 1 record/1 context.

## Câu hỏi mở còn treo

**Chưa xác nhận với C**: cấu trúc chính xác của field `document` (hiện dùng tạm `{"name", "link"}` — C ghi "metadata hoặc parse từ corpus", còn mơ hồ). Các field khác (`document_number`, `article`, `clause`, `text`, `retrieval_score`) đã xác nhận 31/08.
