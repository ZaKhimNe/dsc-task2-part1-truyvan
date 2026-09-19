# Báo cáo thí nghiệm — Truy vấn & Xếp hạng lại (Part 1)

DSC2026 · Nội dung 2 (LegalQA) · Cập nhật 19/09/2026

Tài liệu này ghi lại **mọi thí nghiệm đã chạy** trên phần truy vấn và reranker,
kèm ngưỡng đặt trước, kết quả, và quyết định. Thí nghiệm bị loại được giữ lại
nguyên vẹn — biết hướng nào không đi được cũng đáng như biết hướng nào đi được.

---

## 1. Kiến trúc và chỗ nghẽn

Đường ống có bốn chặng:

| Chặng | Việc | Trạng thái sau đợt thí nghiệm |
|---|---|---|
| 1 — BM25 cấp văn bản | tìm theo từ khoá, rồi nở ra Khoản | **đóng góp 3% vào rổ cuối** |
| 2 — bi-encoder | vector trên 432.473 Khoản | **nhúng kèm tiêu đề Điều: +9,2 điểm hạng 1** |
| 3 — cross-encoder | đọc câu hỏi và Khoản cùng lúc, xếp lại | **có trần ~60%, hạ trọng số xuống 20%** |
| 4 — đóng gói | chọn số đoạn, cắt gọt | **2 đoạn thay vì 3, ngân sách 200 âm tiết** |

Phân rã mất điểm (trần tuyệt đối 1,0):

```
truy hồi làm mất   0,335
sinh câu làm mất   0,171
còn lại            0,494      (LB thật 0,4506, dư 0,043 chưa giải thích)
```

Truy hồi tốn gấp đôi phần sinh câu. Đó là lý do toàn bộ đợt này tập trung vào truy hồi.

Giá trị từng khối trong câu trả lời: Khối 2 (trích luật) 0,587 · Khối 3 (kết luận)
0,313 · Khối 1 (mở đầu) 0,084. Khối 2 chính là thứ phần truy hồi giao ra.

---

## 2. Bảng thí nghiệm

| Mã | Câu hỏi | Ngưỡng đặt trước | Kết quả | Quyết định |
|---|---|---|---|---|
| E1 | Trùng `unit_id` có hại không? | trùng xuyên văn bản > 1% | 7.631 id trùng, **100% cùng một Điều**, 0 xuyên văn bản | không chạm → xuống cuối hàng đợi |
| E3a | Nở đoạn ra cả Điều? | phủ chữ tăng ≥ 10 điểm | 63,5% → 82,5% (rổ thật, k=3, n=472)<br>73,9% → 91,5% (BM25 top-20, n=300) | chạm, nhưng E6 bác — xem dưới |
| E6 | Ngân sách âm tiết bao nhiêu? | tìm cực trị | METEOR đỉnh ở 300; nở cả Điều **thấp hơn** | **phủ chữ là chỉ số sai** |
| E6b | Ba cách chọn Khoản × ngân sách | đỉnh lệch 350 quá ±100 | đỉnh 300/500/400, phẳng 300–500 | giữ 350 (sau đổi, xem §4) |
| E5 | Đọc dẫn chiếu sang Điều khác? | METEOR ≥ +2,0 điểm | +0,0121; **91,9% dẫn chiếu trỏ trong cùng văn bản** | không chạm → loại bản top-50 |
| E4 | Nhúng kèm tiêu đề Điều? | ≥ +2,0 điểm **và** cận dưới CI > 0 | +4,40 / +4,50 / +2,50 ở ba bậc phủ | **ĐẠT — thí nghiệm đầu tiên vượt ngưỡng** |
| T1 | Gold đang ở hạng mấy trong 50? | ngoài 50 ≥ 50% → sửa nguồn ứng viên | ngoài 50 chỉ **9,6% / 13,7%** | nghẽn ở **xếp hạng**, không phải sinh ứng viên |
| T2 | Lỗ cắt 50.000 ký tự | — (đo phơi nhiễm) | BM25 mù **25,39%** kho; 21,89% gold nằm sau chỗ cắt | không vá — xem §3 |
| T5 | Vì sao 75 câu tệ đi sau E4? | — (kiểm giả thuyết) | giả thuyết **bị số liệu bác**, cả hai biến ngược chiều | đóng |
| — | Trộn hai bộ xếp hạng | — | đỉnh **ở giữa**, cao hơn cả hai đầu | **ĐẠT** |

---

## 3. Bốn phát hiện đáng kể nhất

### 3.1 Phủ chữ là chỉ số sai để chọn tham số (E3a → E6)

E3a cho thấy nở đoạn ra cả Điều che được nhiều chữ gold hơn hẳn: 63,5% → 82,5%.
Nhưng khi chấm bằng METEOR — thước thật của cuộc thi — điểm lại **tụt**: đỉnh ở
ngân sách 300 (0,4906) còn nở cả Điều chỉ 0,4533.

Lý do: phủ chữ chỉ đếm phần đúng, không trừ phần dư. Nó **đơn điệu tăng** theo
ngân sách nên không bao giờ có cực trị, tức không dùng để chọn được gì. METEOR có
hình phạt phân mảnh (tối đa cắt đôi điểm) nên có cực trị.

Từ đây mọi lựa chọn tham số chấm bằng METEOR.

### 3.2 Bỏ RRF đã vô hiệu hoá nhánh BM25 — tác dụng phụ không ai ghi lại

`search_units_hybrid` = union(BM25→nở, dense top-300) → **xếp bằng điểm dense** →
cắt 50.

Vì `dense top-300 ⊇ dense top-50` và xếp thuần bằng điểm dense, một unit do BM25
thêm vào chỉ lọt top-50 nếu điểm dense của nó vượt unit thứ 50 — mà nếu vậy nó
đã nằm sẵn trong dense top-50. **Về mặt cấu trúc, BM25 không thể đóng góp gì.**

Đo được:

```
trùng lặp rổ-cuối ↔ dense top-50   96,8% (bare) · 96,2% (title)
unit KHÔNG có trong dense top-50   1,6/50 · 1,9/50
số câu rổ-cuối TRÙNG HOÀN TOÀN     43,8% · 45,9%
```

Quyết định xếp hạng thuần bằng dense thay vì RRF (Hit Khoản@3 55,5% so với 27,0%)
đúng ở thời điểm đó, nhưng nó giết nhánh BM25 như một tác dụng phụ, và điều đó
sống sáu tuần không ai biết.

Hệ quả: lỗ cắt 50.000 ký tự ở T2 — dù BM25 mù một phần tư kho — **không chảy vào
điểm**, vì không ai dùng tới BM25 nữa. Không vá.

### 3.3 Reranker có trần, nâng nhánh yếu và dìm nhánh mạnh

```
nhánh bare  (yếu)   54,9%  →  60,1%     reranker +5,2
nhánh title (mạnh)  64,8%  →  60,7%     reranker −4,1
```

Cả hai rơi về cùng một chỗ. Đó là dấu hiệu reranker **ghi đè** thứ tự nền bằng ý
kiến riêng, và ý kiến riêng của nó đáng đúng khoảng 60%.

Bài học phân biệt hai trục: một bộ xếp hạng lại chỉ sửa **thứ tự**, không sửa
**sự có mặt**. Nên nó vô dụng khi đáp án vắng mặt trong rổ, và có hại khi thứ tự
nền đã tốt hơn nó.

Cách chữa không phải gỡ nó, mà là cho nó một phần tiếng nói.

### 3.4 Xếp hạng tốt lên thì số đoạn tối ưu giảm xuống

Đo trên harness cả bài (Khối 1 và Khối 3 lấy từ gold, chỉ thay Khối 2):

```
ngân sách     k=1       k=2       k=3     dài k=2
    150    0.6714    0.7119    0.6935     376 âm tiết
    200    0.6848    0.7135    0.6865     450   ← đỉnh
    250    0.6935    0.7120    0.6784     525
    350    0.7005    0.7066    0.6648     599
    500    0.7030    0.6983    0.6521     676
```

Quy về **tổng độ dài giao đi** thì cả ba cột cùng chỉ về một chỗ: tối ưu quanh
**450–500 âm tiết**, còn chia thành mấy đoạn là chuyện phụ. Gold Khối 2 median
khoảng 200 âm tiết.

Kết quả này ngược một phép đo cũ (ghép 3 đoạn 0,6651 > ép 1 đoạn 0,6515, biên lợi
k1→k2 +0,0486). Giải thích: phép đo cũ chạy trên rổ có hạng 1 đúng ~39%; rổ hiện
tại đúng 65%. Khi hạng 1 thường đúng thì đoạn thêm vào chỉ pha loãng; khi hạng 1
hay sai thì đoạn thêm vào là cứu cánh.

Phép so một biến (cùng ngân sách, cùng α, cùng nhánh):

```
ngân sách 350:  k=3 0,6648 → k=2 0,7066   +0,0418  CI95 [+0,0365, +0,0466]
ngân sách 200:  k=3 0,6865 → k=2 0,7135   +0,0271  CI95 [+0,0216, +0,0320]
```

---

## 4. Cấu hình chốt

| Nút | Trước | Sau | Căn cứ |
|---|---|---|---|
| Nhúng cái gì | Khoản trần | **+ tiêu đề Điều** | E4, T1 |
| Xếp thứ tự | reranker toàn quyền | **RRF trọng số α = 0,7** | quét α + kiểm chéo hai nửa |
| Số đoạn giao | 3 | **2** | bảng §3.4 |
| Ngân sách mỗi đoạn | 350 | **200** | bảng §3.4 |
| Giới hạn tiêu đề | không | **không** | F2 (+0,0496) đo không có giới hạn |

Công thức trộn:

```
s = α/(60 + dense_rank) + (1−α)/(60 + rerank_rank)
```

Dạng nghịch đảo chứ không trung bình tuyến tính, vì sản xuất chỉ giao 2 đoạn nên
chỉ đỉnh danh sách có giá trị.

**Về α = 0,7:** quét trên mẫu train seed 1909 cho đỉnh phẳng trong dải 0,6–0,8.
Kiểm chéo hai nửa chọn 0,6 và 0,8, cả hai đều thắng α=1,0 trên nửa giữ riêng,
không đảo dấu. Lấy trung điểm 0,7.

Lợi ích ước lượng **bằng kiểm chéo là +0,006** — không phải +0,0082 của bootstrap
toàn mẫu. Chênh lệch giữa hai con số chính là độ thiên lệch do chọn bằng mắt.

---

## 5. Hướng đã đóng — đừng thử lại

| Hướng | Vì sao đóng |
|---|---|
| Nở đoạn ra cả Điều | METEOR tụt, dù phủ chữ tăng 19 điểm |
| Phủ chữ làm chỉ số chọn tham số | đơn điệu tăng, không có cực trị |
| Đọc dẫn chiếu từ top-50 | +0,0121 dưới ngưỡng; thêm ~17 Điều rác mỗi câu |
| Vá lỗ cắt 50.000 ký tự | chỉ ảnh hưởng BM25, mà BM25 đóng góp 3% |
| Sửa trùng `unit_id` | 100% bản sao cùng một Điều; phơi nhiễm thực tế 1,5% |
| Tách từ bigram cho BM25 | không áp dụng — đã dùng `underthesea.word_tokenize` |
| Cap độ dài tiêu đề | F2 đo không có cap; 4,29% vượt 30 âm tiết, 17/1912 nghiêm trọng |

Ghi kèm điều kiện: **E5 đo trên rổ có 47,2% sai Điều.** Nếu rổ tốt lên đáng kể thì
hướng đó là bài toán khác, không nên coi là đóng vĩnh viễn.

---

## 6. Bài học về phương pháp đo

1. **Đếm số ca ≠ mức độ mỗi ca.** Chỗ cắt chạm 10,78% văn bản nhưng cắt tới 0,88%
   nội dung ở ca nặng nhất.
2. **Một comment tự khẳng định con số mà không ai đo thì sống được sáu tuần.**
   Comment "cắt rất ít văn bản" viết 06/08, sai, tồn tại tới khi đo ngày 19/09.
3. **Khi có sẵn dữ liệu để đo thì đừng tranh luận về cách đọc.** `char_start` nằm
   sẵn trong dữ liệu suốt.
4. **Hai công thức trộn khớp nhau không phải bằng chứng độc lập** — chúng ăn cùng
   hai danh sách và đều đơn điệu theo cùng một tham số. Bằng chứng phải là khoảng
   tin cậy hoặc tập giữ riêng.
5. **Đổi một tham số làm tham số khác đổi nghĩa.** Hạ số đoạn từ 3 xuống 2 làm nút
   cắt thật chuyển từ `TARGET_TOTAL` sang ngân sách mỗi đoạn.
6. **Đơn vị đo phải ghi kèm mọi con số.** Cấp Điều ≠ cấp Khoản; ba mốc METEOR khác
   harness (0,42 / 0,49 / 0,63) không đặt cạnh nhau được.
7. **Ngưỡng đặt trước, và được phép để trống khoảng giữa.** Vạch duy nhất bịa ra
   sẽ tự trả lời thay cho suy nghĩ.

---

## 7. Tái lập

Repo **không chứa dữ liệu** — dữ liệu BTC cấp không được tái phát tán (xem
`.gitignore` và `README.md`). Đặt dữ liệu gốc vào `../data/` rồi chạy theo
`README.md`.

Build gói theo cấu hình chốt:

```
python pipeline/build_qa_packages_public_v8_tieu_de_khoan.py \
  --rerank-path outputs/layer3/L3_rerank_public_title.jsonl \
  --alpha 0.7 --max-contexts 2 --per-item 200 --expected-median 450
```

Chạy không tham số thì tái hiện cấu hình cũ. Tên file ra suy từ cấu hình để không
thể dán nhầm nhãn giữa hai bản build khác rổ.
