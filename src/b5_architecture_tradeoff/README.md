# B5 — Đối chứng kiến trúc (phân bổ ngân sách tham số retrieval vs generator)

Nguồn: tài liệu phân công gốc "DSC2026 — Phân công nhóm Task 2" mục 3.3. Ưu tiên **P1**.

## Bài toán

So sánh phương án phân bổ ngân sách tham số — càng nhiều tham số B (retrieval) dùng, C (generator) càng ít.

**CẬP NHẬT 31/08/2026 (chính thức, xác nhận qua C)**: tổng ngân sách toàn hệ thống là **4 tỷ tham số**. C (Generator) dùng **1,7 tỷ**, phần còn lại **2,3 tỷ dành cho B (retrieval)**. (Lịch sử: 13/08 từng ghi nhận tạm "~2 tỷ" qua lời nhắc không chính thức — con số đó đã LỖI THỜI, thay bằng 4 tỷ ở đây.)

## Quyết định ĐÃ CHỐT (13/08/2026, có số liệu thật — không còn là giả thuyết)

Khác bản cũ ("chưa chốt gì, chờ B7") — giờ **đã có số liệu thật từ ablation trên `data_retrieve`**, đủ để quyết định:

**Giữ CẢ 2: BM25 (Layer 1/1.5) VÀ dense (Layer 2, BGE-M3)** — không phải chọn 1. Lý do: đo được BM25 và dense **thất bại ở những câu khác nhau** (chỉ 2.6% cùng thất bại) — bổ trợ nhau thật, không phải dense chỉ "làm lại việc BM25 tốt hơn 1 chút". Hợp (union) 2 nguồn: coverage tăng từ 89.8% (BM25 riêng) lên **97.4%**.

**Ngân sách sử dụng thật**:

| Layer | Model | Tham số |
|---|---|---|
| Layer 1 (BM25) | `rank_bm25` | 0 |
| Layer 1.5 (rerank Khoản) | BM25 (dùng lại) | 0 |
| Layer 2 (bi-encoder) | `BAAI/bge-m3` | 568M |
| Layer 3 (reranker) | `BAAI/bge-reranker-v2-m3` (cùng họ BAAI với Layer 2) | ~568M |
| **Tổng B dùng** | | **~1,136 tỷ / 2,3 tỷ được cấp (49,4%)** |

**~1,164 tỷ còn dư (50,6% phần của B) — QUYẾT ĐỊNH KHÔNG DÙNG HẾT** — đúng tinh thần "chỉ tiêu khi có bằng chứng cần" đã chốt từ đầu, không tiêu chỉ vì "còn ngân sách". Retrieval đã đo được coverage 97,4% (Hit@50), Layer 3 rerank xác nhận cải thiện thêm +10-14đ% Hit@1 — không thấy bằng chứng cần model lớn hơn.

## Kiến trúc cuối (ĐÃ CODE, đang chạy)

```
Câu hỏi → [Layer 1 BM25 ∥ Layer 2 dense] → Union → Expand → Layer 1.5 (rerank Khoản)
        → Layer 3 (reranker, ĐÃ CHỐT GIỮ) → B4 (top-3) → B6 → Generator (C)
```

## Còn treo

- B7 (oracle qua Generator, thuộc trách nhiệm A không phải B) vẫn cần để đo tác động thật lên điểm cuối (METEOR) — B5 ở đây mới dừng ở bằng chứng Recall@K/MRR/Hit@K, chưa phải điểm thi thật.
- Nếu sau này quyết định dùng thêm phần dư 1,164 tỷ (vd fine-tune, model lớn hơn) — cần bằng chứng đo được cụ thể trước, không đổi chỉ vì "có sẵn ngân sách".
