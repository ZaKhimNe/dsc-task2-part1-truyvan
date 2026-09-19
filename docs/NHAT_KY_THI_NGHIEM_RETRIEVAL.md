# Nhật ký thí nghiệm retrieval — MỤC LỤC QUYẾT ĐỊNH

Theo mẫu `PLAN_TOI_THUONG_RETRIEVAL_v2.md` §7. Cập nhật 18/09/2026 22:35.

**Vai của file này:** mỗi thí nghiệm **một dòng**. Không có bảng dài, không có diễn giải.
Bằng chứng, điều kiện đo, script nào sinh ra số nào → [`KET_QUA_THUC_NGHIEM_18-09.md`](KET_QUA_THUC_NGHIEM_18-09.md).

**Cột `nguồn`** — thứ duy nhất phân biệt ba mức tin đang dễ bị trộn:

| Giá trị | Nghĩa |
|---|---|
| `đo gốc` | script sinh ra con số này có trên đĩa, chạy lại được |
| `viết lại rồi khớp` | script gốc mất, đã viết lại và chạy ra cùng số |
| `chỉ còn trong bản ghi` | không có script trên máy này — **đừng trích dẫn như số đã kiểm** |

---

## Mục lục

| Mã | Ngưỡng đặt trước | Kết quả | Rổ | n | Quyết định | Nguồn |
|---|---|---|---|---|---|---|
| **E1** trùng `unit_id` | trùng xuyên văn bản > 1% → E2 lên đầu | 100% bản sao **cùng một Điều**, 0 xuyên văn bản; 95,6% khác text | — (toàn corpus) | 432.473 hàng | **KHÔNG chạm ngưỡng → E2 xuống cuối.** Sửa bằng đánh số duy nhất `uid#2`, KHÔNG lấy max trên danh sách vị trí | `đo gốc` |
| **E3a-heldout** nở ra cả Điều | phủ chữ tăng ≥ 10 điểm | 63,5% → 82,5% (+19,0); ≥90%: 22,2%→49,4%; 354 tốt hơn / 1 xấu hơn | rổ thật, k=3 | **472 câu** | Chạm ngưỡng, nhưng **không** nở cả Điều — xem E6b | `chỉ còn trong bản ghi` |
| **E3a-bm25** nở ra cả Điều | — | 73,9% → 91,5%; một Khoản đủ chỉ 8,7% số câu | BM25 top-20 | 300 câu | Mở rộng sang Điều là đúng hướng | `đo gốc` |
| **E6** quét ngân sách (phủ chữ) | tìm cực trị | đỉnh 300, nhưng hỏng do thiên vị độ dài khi chọn Khoản xuất phát | BM25 top-20 | 300 câu | **Bỏ**, thay bằng E6b | — |
| **E6b** quét ngân sách (METEOR) | đỉnh lệch khỏi 350 quá ±100 → đổi tham số | đỉnh 400–500, vùng phẳng 300–500; cả Điều thua ở **cả ba** bộ chọn | BM25 top-20 | 300 × 3 bộ chọn | **Giữ `PER_ITEM_BUDGET_SYLLABLES=350`. Trục ngân sách ĐÓNG** | `đo gốc` |
| **E5** đọc dẫn chiếu | ≥ +2,0 điểm | +1,7 điểm% (phủ ≥90%); METEOR +0,0121; 8 tốt / 1 xấu / 291 không đổi | BM25 top-20, cắt top-50 | 300 câu | **KHÔNG chạm ngưỡng → loại bản top-50.** Còn đáng thử: chỉ đọc dẫn chiếu từ top-5 | `đo gốc` |
| **E2** vá mã trùng | bật `dedup_by_row` phải hơn nhiễu | rổ mất 0,58 suất/câu (bare) · 1,04 (title), nhưng lấy lại chỉ được METEOR +0,0003 / +0,0014, CI sát 0 | dense thật top-50 | 1.000 × 2 | **Giữ bản vá (hết đè im lặng, rerank dùng `row`), KHÔNG bật cờ. ĐÓNG — có điều kiện: đổi `PER_ITEM_BUDGET_SYLLABLES` thì đo lại** | `đo gốc` |
| **E4** nhúng kèm tiêu đề | recall@50 ≥ +2,0 **và** CI dưới > 0 | phủ ≥80%: 76,4%→**80,8%** (+4,40, CI [+3,00,+5,90]); ≥90%: +4,50; ≥95%: +2,50. METEOR +0,0217 (CI [+0,0142,+0,0291]), 125 tốt/75 tệ | dense thật top-50 (Kaggle) | 1.000 câu | **ĐẠT NGƯỠNG → dùng nhánh title.** Đi tiếp: rerank public rồi V9 | `đo gốc` |

---

## Hai con số E3a KHÔNG so được với nhau

E3a-heldout khởi đầu ở **63,5%**, E3a-bm25 khởi đầu ở **73,9%** — vì 20 đoạn đương nhiên che
nhiều chữ hơn 3 đoạn. Nên **+19,0 và +17,6 không phải "hai lần đo cùng một hiệu ứng ra gần
nhau"**; chúng là hai hiệu ứng trên hai bài toán khác độ khó. Đừng lấy trung bình, đừng coi
cái này xác nhận cái kia.

## E4 — trạng thái nối Kaggle (18/09 22:35)

| | |
|---|---|
| Kernel | `zakhim/dsc-legalqa-e4-title-embedding`, private, GPU T4 |
| Đẩy lúc | 21:24, xác minh `RUNNING` qua API |
| CLI | `retrieval/.venv/Scripts/kaggle.exe` 2.2.4 — **cài trong venv, KHÔNG toàn máy**, nên `which kaggle` rỗng là đúng |
| Token | `C:\Users\GIGA\.kaggle\access_token` (đường tuyệt đối — đừng tra qua `~`, `~` khác nhau giữa Git Bash và WSL/container) |
| Máy | `DESKTOP-UTRKKMD`, Windows 11 + MINGW64 |
| Nhúng bare | xong 2.597s, shape (432.473, 1024) — **432.473 khớp E1** |
| Bảo mật | token đã từng dán vào khung chat → **thu hồi và tạo lại** ở kaggle.com/settings |

## Luật chia file khi hai phiên chạy song song

| Phiên | Giữ file nào |
|---|---|
| Phiên có Kaggle (máy `DESKTOP-UTRKKMD`) | `pipeline/eval_*`, `pipeline/measure_dup_unit_id_damage.py`, **cả hai file `docs/` này** |
| Phiên kia | `src/b6_context_package/format_unit.py`, `src/b6_context_package/package.py`, `pipeline/eval_quotation_format_gap.py`, `pipeline/build_qa_packages_public_v8_tieu_de_khoan.py` |

Chia theo **file**, không chia theo việc. Muốn ghi ra ngoài phần của mình thì báo trước.

## Còn thiếu từ plan v2

P4 reranker có tiêu đề · P5 quét rổ · P10 script chuẩn hoá · `data123/` ·
`b0_labels_train_sample.json` · `bm25_khoan_index.pkl`.
