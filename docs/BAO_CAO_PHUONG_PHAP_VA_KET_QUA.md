# DSC 2026 — Phương pháp đo và kết quả

**Hai nội dung thi:** LegalIR (truy hồi văn bản pháp luật, recall@5) · LegalQA (sinh câu
trả lời từ văn bản luật, METEOR).

**Tài liệu này viết cho người phản biện.** Nó không kể lại quá trình. Nhật ký chi tiết
theo thời gian nằm ở [`BAO_CAO_TONG_HOP_THI_NGHIEM.md`](BAO_CAO_TONG_HOP_THI_NGHIEM.md);
điều kiện đo và script sinh ra từng con số nằm ở `QA/part1-truy-xuat/retrieval/docs/`.
Hai dạng tài liệu được giữ tách rời có chủ ý: gộp chúng sẽ làm mất phần **điều kiện đo**
đi kèm mỗi con số, mà đó chính là thứ quyết định con số có đọc được hay không.

---

## 1. Đóng góp chính không phải điểm số

Điểm cuối: LegalIR **0,9124** (public LB) · LegalQA **0,5383** (private LB).

Thứ đáng đưa ra phản biện là **kỷ luật đo**. Trong đợt thí nghiệm cuối, **bốn kết quả có
lợi cho chính nhóm đã bị chính nhóm bác bỏ**, mỗi lần bằng một cơ chế kiểm khác nhau:

| Thí nghiệm | Con số đầu tiên | Cơ chế kiểm | Con số sau khi kiểm | Quyết định |
|---|---:|---|---:|---|
| **T9** — chấm lại trong cửa sổ top-W | +0,0019 (bootstrap toàn mẫu) | kiểm chéo hai nửa | **−0,0002** | đóng |
| **T10** — viết Khối 3 khi gold không có | −0,0508 (nhóm cần đo) | **nhóm đối chứng** | phần riêng chỉ **−0,0059** | đóng |
| **P2** — fine-tune bi-encoder mức Khoản | recall@5 **+2,96 điểm** | ngưỡng đặt trước trên recall@50 | **−0,23 điểm** | đóng |
| **Khối 3 rỗng** | 12,1% ("còn nhiều dư địa") | đối chiếu **danh tính lượt chạy** | **0,9%** trên bài đã nộp | không sửa |

Ba trong bốn trường hợp, con số đầu tiên đủ đẹp để báo cáo được. Không con số nào sống
sót qua phép kiểm thứ hai.

Một báo cáo chỉ gồm số đẹp không cho phản biện cách nào kiểm tra nó. Bốn dòng trên là
bằng chứng rằng bộ lọc có hoạt động.

---

## 2. Tám quy tắc đo, và cái giá đã trả cho từng cái

Mỗi quy tắc dưới đây ra đời từ một lần mắc lỗi có thể định lượng.

### 2.1 Ngưỡng phải đặt trước, và phải tương đối so với một mốc đã có

Không bao giờ là con số tuyệt đối bốc ra. Đã sai bốn lần vì vi phạm điều này — lần gần
nhất là đặt cổng "bộ phân loại chỉ dùng độ dài phải dưới 65%", trong khi **mức tự nhiên
của kho đo trên 18.177 cặp là 74,2%**. Cổng 65% ngầm giả định mốc đúng là 50%, và nó suýt
loại một tập dữ liệu hợp lệ.

### 2.2 Mọi phép đo phải in kèm một dòng đối chứng

T10 là ví dụ đắt nhất: đo riêng nhóm cần quan tâm cho **−0,0508**; thêm nhóm đối chứng
cho **−0,0449**. Phần riêng của hiệu ứng cần đo chỉ là **−0,0059** — **nhỏ hơn 8,6 lần**
con số đầu tiên. Chi phí của dòng đối chứng: 4 dòng mã.

Ở Task 1, dạng cụ thể của quy tắc này là: mọi bảng đánh giá bộ chấm lại phải có dòng
*"rổ trần, không reranker"*. Dòng đó đã cứu 3 giờ GPU và cả một nhánh nghiên cứu.

### 2.3 Lợi ích báo cáo bằng kiểm chéo, không bằng bootstrap toàn mẫu

Khi một tham số được **chọn** trên chính dữ liệu dùng để chấm, bootstrap toàn mẫu đo cả
phần thiên lệch do chọn.

| Thí nghiệm | Bootstrap toàn mẫu | Kiểm chéo hai nửa | Thiên lệch |
|---|---:|---:|---:|
| quét α (11 giá trị) | +0,0082 | +0,006 | +0,0022 |
| T9 — quét W × α (24 ô) | +0,0019 | **−0,0002** | **+0,0021** |

Ở T9, **thiên lệch lớn hơn chính hiệu ứng**. Hai nửa còn không đồng ý về tham số nào tốt:
nửa 1 chọn α=0,7 (ra +0,0014), nửa 2 chọn α=0,5 (ra −0,0018).

### 2.4 Một con số không mang danh tính lượt chạy nguy hiểm ngang một file không có md5

Dự án này gắn md5 cho mọi artefact — file rerank, gói câu hỏi, mã nguồn đã vá — đúng để
không bao giờ so nhầm hai thứ tưởng là một. Quy tắc đó **phải áp cho cả con số trích từ
nhật ký**.

Ca cụ thể, 21/09: ba mẩu dữ kiện đều đúng riêng lẻ — bản vá `4c1cde4` nằm trong HEAD,
`configs/models.json` vẫn 256/160, và Khối 3 rỗng 12,1% — được ghép thành một cơ chế
nghe xuôi: *"cơ chế thử lại đã sửa nhưng ngân sách token vẫn thiếu"*. Ba mẩu đến từ **hai
lượt chạy khác nhau**.

Thứ làm chuỗi suy luận đó có vẻ tự nhất quán là chỉ số `retry_used` **36,8%** — mà nó
bằng nhau ở **cả hai** lượt, tức là chỉ số **không mang thông tin phân biệt**. Chỉ số
thật sự phân biệt là `token_limit_retry_used`: 8,6% → 20,0%.

Cách bắt: bảng ba nguồn, hai cách đếm độc lập.

| | `conclusion` rỗng | `answer` thiếu Khối 3 | `retry_used` | `token_limit_retry` |
|---|---:|---:|---:|---:|
| V9 train, code **cũ** | 12,1% | 14,2% | 36,8% | 8,6% |
| V9 train, code **mới** | **1,4%** | 3,5% | 36,8% | 20,0% |
| **private — bài đã nộp** | **0,9%** | 2,2% | 40,2% | 21,9% |

Không bảng một-nguồn nào bắt được điều này.

### 2.5 Đo trần trước khi đầu tư

Ba hướng bị đóng ở Task 1 chỉ bằng phép đo trần trên CPU, trước khi tiêu giờ GPU: `M=30`
(~4h), chưng cất từ model 8B (~18h), chọn tham số thích nghi theo câu. Tổng hệ thống cổng
đặt trước cứu **≈21 giờ GPU**.

### 2.6 Chỉ số phụ đơn điệu không chọn được tham số — và cũng không chọn được nhãn

**Độ phủ chữ** tăng đơn điệu theo độ dài ứng viên, nên không có cực trị. Dùng nó để chọn
ngân sách đoạn cho kết quả sai (E6), và điều đó đã được ghi thành bài học.

Cùng lỗi tái xuất ở chỗ khác ba tuần sau: dùng độ phủ để **chọn nhãn** cho dữ liệu huấn
luyện thì "Khoản phủ nhiều chữ nhất" hoá ra là "Khoản **dài nhất**".

| | nhãn chọn bằng recall | nhãn chọn bằng **F1** |
|---|---:|---:|
| độ dài Khoản, trung vị | 334 âm tiết | **119** |
| Khoản khác nhau / số cặp | 79% | **95%** |
| một Khoản bị gán cho bao nhiêu câu, tối đa | **38** | 5 |
| bộ phân loại **chỉ dùng độ dài** đoán đúng | **94,3%** | 68,6% (tự nhiên: 74,2%) |

Một bài học đã trả tiền vẫn tái phát khi không được phát biểu đủ tổng quát.

### 2.7 Khi can thiệp làm dịch tối ưu của tham số đi kèm, ngưỡng phải đặt ở tối ưu **chọn lại**

Ca cụ thể, T12. Bản đăng ký trước có hai mệnh đề mâu thuẫn nhau, viết trong cùng một lần:

1. *"Ngưỡng chính: METEOR tại cấu hình nộp, **α = 0,7 cố định** ở cả hai nhánh."*
2. *"Nếu bộ chấm lại giỏi lên thì **α tối ưu sẽ dịch xuống** — tin nó nhiều hơn."*

Tức đã **dự đoán trước** rằng α sẽ dịch, rồi vẫn khoá ngưỡng ở giá trị cũ. Ngưỡng 1 do đó
đo bộ chấm lại **mới** tại điểm được tối ưu cho bộ chấm lại **cũ**. Nó trượt một phần
chính vì điều đã được dự đoán.

Số liệu cho thấy đúng như vậy: hiệu ứng +0,0049 tại α=0,7 nhưng +0,0121 tại α=0,5.

> **Quy tắc:** khi một can thiệp được dự kiến làm dịch tối ưu của một tham số đi kèm,
> ngưỡng phải đặt ở **tối ưu chọn lại** — chọn trên nửa A, đo trên nửa B — chứ không đặt ở
> giá trị cũ. Đóng băng tham số đi kèm biến phép đo thành phép đo *"can thiệp này có hoạt
> động **mà không cần chỉnh gì khác** không"*, một câu hỏi khác và khắt khe hơn câu hỏi
> đang cần trả lời.

Ngưỡng 3 của bản đăng ký chính là thiết kế đúng. Bài học là nó lẽ ra phải là ngưỡng
**chính**, không phải ngưỡng phụ.

### 2.8 Cỡ mẫu phải suy từ ngưỡng, không phải ngược lại

P2 đặt ngưỡng +2,0 điểm recall@50 nhưng chỉ có n=439, cho CI nửa-rộng ±2,7 điểm. Nếu hiệu
ứng thật đúng bằng ngưỡng thì CI sẽ là [−0,7; +4,7] — **chứa 0**, tức phép đo không đủ
phân giải để xác nhận chính ngưỡng nó phải kiểm.

Ngoại suy từ CI đã đo (nửa-rộng tỷ lệ nghịch với √n):

| Mục tiêu | n cần | CI nửa-rộng |
|---|---:|---:|
| sàn: CI loại trừ 0 khi hiệu ứng = +2,0 | **≈ 830** | 2,0 |
| thoải mái: CI [+0,5; +3,5] | **≈ 1.500** | 1,5 |

*(Công thức tỷ lệ không ghép cặp `√(p(1−p)/n)` cho ≈1.500–2.000; nó **cao hơn thực tế**
vì phép so ghép cặp — chỉ câu bất đồng mang thông tin.)*

Kết luận của P2 vẫn vững vì quan sát được là **−0,23**, cách ngưỡng đủ xa. Chỗ chết là
nếu nó rơi vào +1…+3.

---

### 2.9 Kiểm bằng thứ phân biệt được đúng với sai, không bằng thứ chỉ xác nhận là nó chạy

Khi dựng kernel T12 bằng script sửa kernel cũ, ký tự `\n` trong chuỗi bị nuốt mất một tầng
escape và biến thành xuống dòng thật ở **năm chỗ** — trong đó có chính dấu phân cách giữa
tiêu đề và Khoản, tức biến của thí nghiệm. **Ba trong năm chỗ vẫn là cú pháp hợp lệ**, nên
kernel chạy được và sẽ cho ra kết quả sai âm thầm.

"Chạy được" không phân biệt được đúng với sai. Cái bắt được lỗi là một phép kiểm nhắm đúng
thứ có thể sai: `ast.parse` cộng một `assert` so dấu phân cách với `chr(92)+chr(110)` sau
khi sửa. Cùng họ với ba phép kiểm khác trong kernel T12: nhánh đối chứng phải có **0**
cặp đổi chữ, nhánh thí nghiệm phải có tỉ lệ đổi chữ trên sàn đặt trước, và ở trọng số
rerank bằng 0 hai nhánh phải trùng **đúng tới chữ số cuối**.

Cùng họ với bẫy Task 1 đã dính — kiểm cache vector bằng số hàng, trong khi vector của đoạn
900 và 1.800 ký tự có cùng số hàng.

---

## 3. Hệ thống

### 3.1 LegalIR — hai chặng

```
kho 8.532 văn bản
   │
 Part 1  cắt đoạn → BM25 ⊕ bi-encoder → hoà RRF → top-50/câu
   │
 Part 2  cross-encoder + đọc sâu → chốt top-5
   ▼
 5 mã văn bản
```

Cấu hình chốt: rổ `_matchEmbedded` · `M=20` · `K=50` · `MERGE_CHARS=1800` ·
`blend n_bm25=2` · `AITeamVN/Vietnamese_Reranker` (0,568B).

### 3.2 LegalQA — bốn chặng, câu trả lời ba khối

```
BM25 cấp văn bản → nở ra Khoản → bi-encoder trên 432.473 Khoản
  → cross-encoder → đóng gói → Qwen3-1.7B + QLoRA V8.3
```

Cấu trúc câu trả lời và tỷ trọng điểm — số này chi phối mọi ưu tiên:

| Khối | Nội dung | Ai sinh | Điểm mất nếu bỏ hẳn |
|---|---|---|---:|
| Khối 2 | trích nguyên văn điều luật | **chương trình chép** từ context | **0,5868** |
| Khối 3 | kết luận | model | 0,3129 |
| Khối 1 | câu dẫn căn cứ | model | 0,0841 |

Khối nặng nhất **không do model sinh ra** — nó là đầu ra của khâu truy hồi. Đó là căn cứ
để dồn công vào truy hồi thay vì vào model.

Cấu hình chốt: nhúng Khoản **kèm tiêu đề Điều** · trộn hạng nghịch đảo α=0,7 · giao 2
đoạn · ngân sách 200 âm tiết/đoạn.

```
s = α/(60 + hạng_dense) + (1−α)/(60 + hạng_rerank)
```

---

## 4. Kết quả

### 4.1 LegalIR — chuỗi tăng tiến trên public LB

| Bước | public LB | Δ |
|---|---:|---:|
| rerank AITeamVN, n=0 | 0,8350 | — |
| + giữ chỗ cho rổ (`n=2`) | 0,8573 | +2,23 |
| + đọc sâu M=20/K=20 | 0,8840 | +2,67 |
| + BM25 chọn đoạn, gộp 1800 ký tự | 0,8871 | +0,31 |
| + rổ hoà RRF (M=10, n=3) | 0,9063 | +1,92 |
| + M=20, n=2 | 0,9109 | +0,46 |
| + rổ `_matchEmbedded`, n=3 | 0,9118 | +0,09 |
| + `K=50`, n=2 | **0,9124** | +0,06 |

Hai đòn lớn nhất — đổi bộ chấm lại và đổi rổ ứng viên — đều là **đổi thứ nạp vào**, không
phải tinh chỉnh cái đang có.

### 4.2 LegalIR — bảng đối chứng 10 bộ chấm chéo

Giao thức: 300 câu dev khoá cứng · **15.000 cặp giống hệt nhau cho mỗi model** · chấm một
tầng · T4. Trần của rổ (recall@50) = 0,9817.

| Model | Nền | Tham số | R@1 | R@5 | MRR@10 | giây/cặp |
|---|---|---:|---:|---:|---:|---:|
| **AITeamVN/Vietnamese_Reranker** | XLM-R, tinh chỉnh tiếng Việt | 0,568B | **0,5017** | 0,7467 | **0,6178** | 0,058 |
| BAAI/bge-reranker-v2-m3 | XLM-R đa ngữ | 0,568B | 0,4683 | **0,7517** | 0,5994 | 0,056 |
| jina-reranker-v2-base-multilingual | XLM-R đa ngữ | 0,278B | 0,4383 | 0,7233 | 0,5677 | 0,025 |
| mmarco-mMiniLMv2-L12 | mMiniLM đa ngữ | 0,118B | 0,4867 | 0,7217 | 0,5963 | **0,007** |
| Qwen3-Reranker-0.6B | Qwen3 **decoder** | 0,596B | 0,4133 | 0,7217 | 0,5627 | 0,257 |
| itdainb/PhoRanker | PhoBERT | 0,135B | 0,4283 | 0,7200 | 0,5549 | 0,014 |
| namdp-ptit/ViRanker | XLM-R tiếng Việt | 0,568B | 0,3917 | 0,6867 | 0,5182 | 0,106 |
| BAAI/bge-reranker-base | XLM-R base | 0,278B | 0,3583 | 0,6850 | 0,5064 | 0,018 |
| ms-marco-MiniLM-L6-v2 *(đối chứng ÂM)* | BERT tiếng Anh | 0,023B | 0,3233 | 0,5633 | 0,4421 | 0,005 |

Ba kết luận:

- **Số tham số không dự đoán chất lượng.** `mMiniLMv2` 0,118B hơn `ViRanker` 0,568B
  **+3,50 điểm** R@5, dù nhỏ hơn 4,8 lần và nhanh hơn 16 lần. Thứ quyết định là dữ liệu
  huấn luyện đúng miền.
- **Kiến trúc decoder không tự động tốt hơn.** `Qwen3-Reranker-0.6B` đạt đúng 0,7217 —
  **bằng** `mMiniLMv2` 0,118B — nhưng chậm hơn 40 lần.
- **Đối chứng âm hoạt động đúng:** model tiếng Anh thuần rơi 18,8 điểm dưới nhóm dẫn đầu,
  xác nhận bảng đo đúng năng lực tiếng Việt chứ không đo nhiễu.

Một phát hiện phụ có giá trị cho cộng đồng: `thanhtantran/Vietnamese_Reranker` cho điểm
**trùng khít từng bit** với `AITeamVN/Vietnamese_Reranker` trên cả 15.000 cặp. Khác tài
khoản HuggingFace, cùng trọng số. Đã loại khỏi bảng để không tính hai lần.

### 4.3 LegalQA — phân rã mất điểm

Phương pháp: thay model bằng **generator hoàn hảo** (lấy Khối 1 và Khối 3 thẳng từ đáp án
chuẩn, chỉ đổi Khối 2). Ghép lại ba khối gốc cho đúng METEOR = 1,0000 nên phép tách không
mất chữ.

| Khối 2 dùng gì | METEOR |
|---|---:|
| oracle context | 1,0000 |
| truy xuất thật, top-3 | 0,6651 |
| truy xuất thật, top-1 | 0,6038 |
| bỏ hẳn Khối 2 | 0,4132 |

| Nguồn mất điểm | Mất |
|---|---:|
| generator (1,0 → 0,8289) | 0,171 |
| **truy xuất** (1,0 → 0,6651) | **0,335** |
| dự đoán LB | 0,494 |
| LB thật của bản V1 | 0,4506 |

Sai số 0,043 — hai nguồn giải thích gần hết khoảng cách. **Truy xuất tốn gấp đôi
generator.**

### 4.4 LegalQA — cấu hình cuối so với cấu hình trước

Đo trên 1.000 câu train có đáp án chuẩn, ghép cặp, bootstrap 2.000 lượt seed 42:

| Cấu hình | METEOR |
|---|---:|
| trước — Khoản trần · α=0 · 3 đoạn · 350 âm tiết | 0,5200 |
| sau — **+ tiêu đề Điều · α=0,7 · 2 đoạn · 200 âm tiết** | **0,5537** |
| **chênh lệch** | **+0,0337** CI95 [+0,0245; +0,0431] |

Private LB: **0,5383**.

**Bộ chấm nội bộ đã được xác minh trùng khít bộ chấm BTC.** `scoring.py` của BTC gọi
`meteor_score(...)` với tham số mặc định của NLTK (α=0,9 · β=3,0 · γ=0,5), và hàm
`build_in_tokenizer` là hàm rỗng (pyvi bị chú thích). Chạy đúng biểu thức của BTC trên
cùng dữ liệu cho **0,553701**, trùng từng chữ số. Do đó 0,5537 (nội bộ) và 0,5383 (LB)
nằm cùng một thang.

---

## 5. Trần đo được, và dư địa còn lại

### 5.1 Nút thắt nằm ở xếp hạng, không ở sinh ứng viên

| Bằng chứng | Số |
|---|---|
| gold Điều nằm **ngoài** rổ top-50 | chỉ 9,6% / 13,7% |
| bộ chấm lại trên nhánh **mạnh** | 64,8% → **60,7%** (làm hại) |
| bộ chấm lại trên nhánh **yếu** | 54,9% → **60,1%** (giúp) |

Hai nhánh rơi về cùng một mức ~60%. Đó là dấu hiệu bộ chấm lại **ghi đè** thứ tự nền bằng
ý kiến riêng, và ý kiến riêng của nó đúng khoảng 60%. Một bộ xếp hạng lại chỉ sửa **thứ
tự**, không sửa **sự có mặt**: nó vô dụng khi đáp án vắng mặt trong rổ, và có hại khi thứ
tự nền đã tốt hơn nó.

Kết luận này được tìm ra **độc lập ở cả hai nội dung thi**. Ở LegalIR, dạng của nó là: rổ
trần @5 = 0,9117 **cao hơn** tầng chấm lại đầy đủ = 0,9100 — chấm lại 50 ứng viên bằng
cross-encoder 568 triệu tham số ra **kém hơn không chấm gì**.

### 5.2 Trần của khâu xếp hạng, và điều nó không nói

Cùng rổ ứng viên, cùng ngân sách, cùng số đoạn — **chỉ đổi thứ tự**:

| Mốc (harness cả bài, n=1.000) | METEOR | So sản xuất |
|---|---:|---|
| sản xuất | 0,7136 | — |
| chọn 2 tốt nhất trong **top-10** | 0,7958 | **+0,0822** CI95 [+0,0758; +0,0888] |
| chọn 2 tốt nhất trong **top-50** | 0,8295 | **+0,1159** CI95 [+0,1083; +0,1235] |

Ba điều phải đọc kèm, nếu không sẽ dùng sai con số này:

1. **Đây là thang harness, không phải thang bảng xếp hạng.** Harness giữ Khối 1 và Khối 3
   ở mức gold nên phóng đại hiệu ứng của Khối 2: cùng cấu hình, harness cho 0,7136 còn LB
   private thật là 0,5383. **Không đặt +0,1159 cạnh +0,0337.** Không có hệ số quy đổi.
2. **Trần oracle không hệ thống nào với tới được** — nó chọn bằng cách nhìn đáp án chuẩn.
3. **Tham lam là cận dưới:** suất thứ hai chọn sau khi cố định suất thứ nhất, không vét
   hết 1.225 cặp.

**Và trần này không nói rằng hai tín hiệu hiện có chứa thông tin để chạm tới nó.** T9 đã
chứng minh vế đó sai: xáo lại bằng chính điểm dense và điểm rerank, ở **mọi** cửa sổ từ 5
đến 50, cho +0,0019 toàn mẫu và −0,0002 khi kiểm chéo. Lý do cơ học: với trọng số nghịch
đảo hạng, một ứng viên ở hạng dense 40 **không thể** bị bộ chấm lại kéo lên đỉnh — α=0,7
đã ghìm sẵn nó, nên cắt cửa sổ W làm lại đúng việc α đã làm.

### 5.3 T12 — hướng duy nhất cho hiệu ứng thật: reranker đọc tiêu đề Điều

E4 đã chứng minh tiêu đề Điều giúp **bi-encoder** phân biệt các Khoản gần giống nhau. Câu
hỏi chưa ai hỏi: **bộ chấm lại có được đọc tiêu đề không?**

Kiểm mã nguồn cả ba kernel rerank (public, train, private): tất cả đều tạo cặp bằng
`pairs.append([question, u["text"]])`, và chuỗi `dieu_tieu_de` không xuất hiện trong file
nào. Nhánh "title" của thí nghiệm trước chỉ đổi **bộ nhúng dense**. **Bộ chấm lại chưa bao
giờ thấy tiêu đề.**

Thiết kế: một rổ (dense-title, cấu hình sản xuất), một lần chạy, hai nhánh khác nhau đúng
chữ đưa vào bộ chấm lại. Mẫu mới 1.000 câu seed 2109, **giao với mẫu đã dùng để chỉnh
tham số = 0**.

| α | reranker đọc Khoản trần | reranker đọc **tiêu đề + Khoản** | chênh lệch |
|---:|---:|---:|---:|
| 0,0 | 0,6822 | 0,7036 | **+0,0214** |
| 0,3 | 0,6911 | 0,7076 | +0,0164 |
| 0,5 | 0,6958 | **0,7079** | +0,0121 |
| 0,7 ← sản xuất | 0,6935 | 0,6984 | +0,0049 |
| 0,9 | 0,6888 | 0,6884 | −0,0004 |
| 1,0 | 0,6841 | 0,6841 | **0,0000** |

Chênh lệch giảm **đơn điệu** theo α, và bằng **đúng 0,0000** ở α=1,0 — nơi bộ chấm lại
không được dùng nên hai nhánh buộc phải trùng. Đó là phép tự kiểm nội tại: nếu biến bị rò
sang nhánh đối chứng thì dòng đó đã khác 0.

Tách bạch từng phần (ghép cặp, bootstrap 2.000 lượt):

| Thay đổi | Δ | CI95 |
|---|---:|---|
| α 0,7 → 0,5, **giữ** bộ chấm lại đọc Khoản trần | +0,0023 | [−0,0017; +0,0067] |
| **tiêu đề, tại α=0,5** — một biến | **+0,0121** | [+0,0072; +0,0169] |
| **tiêu đề, tại α=0,7** — một biến | **+0,0049** | [+0,0014; +0,0085] |
| tổng: sản xuất → tiêu đề + α=0,5 | +0,0144 | [+0,0100; +0,0190] |

**Chỉnh α một mình không mua được gì** (CI chứa 0). Toàn bộ lợi ích đến từ tiêu đề, và độ
lớn của nó phụ thuộc α: càng tin bộ chấm lại thì tiêu đề càng có giá. Bộ chấm lại xếp khác
nhau ở **1.000/1.000** câu, nên hiệu ứng không đến từ vài câu lẻ.

> **Ngưỡng đặt trước đã TRƯỢT, và điều đó được ghi lại nguyên vẹn.** Ngưỡng 1 là *"METEOR
> tại cấu hình nộp (α=0,7) ≥ +0,010"*; kết quả **+0,0049**. Cái đạt ngưỡng là ở α=0,5
> (+0,0121) và kiểm chéo chọn α trên nửa giữ riêng (+0,0131) — tức một cấu hình **đã đổi
> hai thứ**.
>
> Bản đăng ký trước có mâu thuẫn nội tại do chính chúng tôi gây ra: ngưỡng 1 đóng băng α ở
> giá trị được tối ưu cho bộ chấm lại **cũ**, trong khi ngưỡng 3 tồn tại chính vì đã dự
> đoán α tối ưu sẽ dịch. Hai ngưỡng đo hai cấu hình triển khai khác nhau và bản đăng ký
> không nói cái nào quyết định.
>
> Hiệu ứng là **thật** (CI loại trừ 0 ở cả hai mức α, có cơ chế, có phép tự kiểm ở α=1,0).
> Nhưng mọi quyết định triển khai dựa trên nó là **phán đoán đưa ra sau khi thấy số**, chứ
> không phải "đạt ngưỡng".

**Trạng thái triển khai** (ghi 21/09/2026, khoảng 04:35 UTC). Quyết định triển khai một
lượt private với reranker đọc tiêu đề và α = 0,5, **dù ngưỡng 1 trượt**. Lý do: hiệu ứng
có chiều rõ và có cơ chế, và theo Điều 5 bản tốt nhất được tính nên lượt này không thể
làm mất 0,5383. Kernel rổ private đã dựng
(`kaggle_layer3/rerank_private_title_input_notebook.py`). **Điểm LB của lượt này chưa có
tại thời điểm ghi.**

Lượt này đổi **đúng hai thứ** so với bài 0,5383: chữ đưa vào reranker, và α từ 0,7 xuống
0,5. Kỳ vọng trên LB, đặt trước khi có số: khoảng **+0,003 đến +0,012**. Cơ sở là cặp hiệu
chỉnh duy nhất có được giữa harness và end-to-end — V8→V9, harness khoảng +0,049, end-to-end
trên train +0,0337, tỉ lệ khoảng 0,7 — áp vào +0,0144. Cặp đó không phải phép so một biến
sạch nên đây là bậc độ lớn, không phải dự đoán. Quanh 0 trên LB nghĩa là harness không
chuyển được; không mất gì.

Chưa đo: số cặp vượt giới hạn 512 token của reranker ở mỗi nhánh (tiêu đề làm chuỗi dài
thêm, và cắt từ bên phải là mất đuôi Khoản).

### 5.4 Phát biểu kết luận cho đúng

Kết luận ở mục này **viết trước T12 và đã phải thu hẹp phạm vi**. Nó áp cho hai thứ: xáo
lại bằng hai tín hiệu sẵn có (T9), và fine-tune bi-encoder mức Khoản — đúng can thiệp đã
mang lại **+1,90 điểm bảng thật** cho LegalIR — vốn không vượt ngưỡng:

| | gốc | fine-tune | Δ | CI95 |
|---|---:|---:|---:|---|
| recall@1 | 0,2301 | 0,2346 | +0,46 điểm | [−2,96; +3,87] |
| recall@5 | 0,4806 | 0,5103 | +2,96 điểm | [−0,68; +6,61] |
| **recall@50** | **0,7403** | **0,7380** | **−0,23 điểm** | [−2,96; +2,51] |

Ngưỡng đặt trước: recall@50 ≥ +2,0 **và** cận dưới CI > 0. Trượt cả hai.

**Quy ra số câu cho dễ thấy.** Với n=439, mỗi câu đổi kết quả đáng **0,228 điểm phần
trăm**. Nên `+0,46` trong bảng là **hai câu**; `+2,96` là mười ba câu; và ngưỡng `+2,0`
mà chúng tôi tự đặt trước là **chín câu trên bốn trăm ba mươi chín**. Một phép đo phân
biệt được chín câu là một phép đo mỏng — đó chính là nội dung của §2.8, phát biểu lại
bằng đơn vị đếm được thay vì bằng khoảng tin cậy (§2.8).

> **Với các tín hiệu đã thử — xáo lại dense và reranker cũ, và bi-encoder fine-tune — đóng
> góp biên đã tụt xuống dưới ngưỡng phát hiện của dữ liệu hiện có, nên đóng các hướng đó.**
>
> Khâu xếp hạng **nói chung** thì chưa cạn: T12 (§5.3) cho thấy một tín hiệu tốt hơn —
> reranker được đọc tiêu đề Điều — chuyển được thành điểm đo được, CI loại trừ 0. Nên
> phát biểu đúng là *"các tín hiệu đã thử không còn gì"*, không phải *"khâu xếp hạng
> không còn gì"*.

Đây **không** phải phát biểu "mô hình không còn tín hiệu để khai thác" — không có bằng
chứng cho câu đó. Bảng cỡ mẫu ở §2.8 nói chính xác cần bao nhiêu dữ liệu để nhìn xa hơn:
**n ≈ 830** là sàn, **n ≈ 1.500** là thoải mái, và tập train có 6.000 câu chưa dùng.

---

### 5.5 Hướng còn mở, đã lập kế hoạch nhưng chưa chạy

Hai hướng dùng **mô hình đã đăng ký, không huấn luyện gì**, nên không chạm Điều 4. Kế hoạch
chi tiết, ngưỡng đặt trước và mẫu báo cáo ở
[`KE_HOACH_THI_NGHIEM_D_E.md`](KE_HOACH_THI_NGHIEM_D_E.md).

| | Hướng | Nhắm vào | Kỳ vọng |
|---|---|---|---|
| **D** | Qwen3-1.7B chấm từng cặp trong top-10, trộn như tín hiệu thứ ba (hướng của RankGPT, Sun và cộng sự 2023) | thứ tự trong top-10 — nơi T6 đo được 71% dư địa | thấp — model nhỏ — nhưng là tín hiệu **mới**, đúng thứ T9 cho thấy đang thiếu |
| **E** | HyDE: nhúng một đoạn luật giả do model viết (Gao và cộng sự 2022) | sự có mặt trong rổ | thấp hơn D — sự có mặt không phải nút thắt (T1); có cửa rẻ trước |

Cả hai đo trên mẫu mới seed 2209, không giao với các mẫu đã dùng để chỉnh tham số.

---

## 6. Hướng đã đóng

Giữ nguyên vì biết hướng nào không đi được đáng bằng biết hướng nào đi được. Mỗi dòng là
một phép đo đã trả tiền.

### 6.1 LegalIR

| Hướng | Kết quả |
|---|---|
| fine-tune toàn phần (2 lần) | **−0,50** rồi **−6,50** |
| chưng cất từ Qwen3-8B | NET −16 nhóm/600; **thầy 0,5417 thua trò 0,5683** |
| học lại đầu phân loại (encoder đóng băng) | Δ tốt nhất +0,67; **18 câu sai: 5/18 → 5/18, không đổi câu nào** |
| ensemble nhiều bộ chấm | âm |
| gộp điểm z-score / RRF hai thứ hạng | thua giữ-chỗ ở **mọi** tham số |
| ưu thế theo loại văn bản / thẩm quyền | âm, kể cả khi ước tham số ngay trên tập dev |
| ưu tiên văn bản mới / còn hiệu lực | gold là văn bản mới nhất đúng **2,0%** = ngẫu nhiên |
| ghép tên văn bản | slug không dấu −2,0; tiêu đề có dấu +1,00 nhưng **83% trùng** tầng 2 |

### 6.2 LegalQA

| Hướng | Vì sao đóng |
|---|---|
| nở đoạn ra cả Điều | METEOR tụt dù phủ chữ tăng 19 điểm |
| phủ chữ làm chỉ số chọn tham số | đơn điệu tăng, không có cực trị |
| đọc dẫn chiếu từ top-50 | +0,0121 dưới ngưỡng; thêm ~17 Điều rác mỗi câu |
| vá lỗ cắt 50.000 ký tự trong BM25 | chỉ ảnh hưởng BM25, mà BM25 **đóng góp 3%** vào rổ cuối |
| sửa trùng `unit_id` | 100% bản sao cùng một Điều; phơi nhiễm thực tế 1,5% |
| few-shot toàn cục | METEOR −0,0187; chỉ nhóm trình tự/danh sách tăng |
| **nới ngân sách token cho Khối 3** | **hai nguồn độc lập cùng dấu âm** — xem dưới |
| chấm lại trong cửa sổ top-W | T9, trượt cả ba ngưỡng |
| fine-tune bi-encoder mức Khoản | P2, trượt ngưỡng |

**Về nới ngân sách token**, vì nó trực giác nhất và đã bị đề xuất lại nhiều lần:

```
thí nghiệm V5 (bỏ giới hạn, 256→512, retry kết luận 384):
    METEOR Khối 3   0,3965 → 0,4752   (+0,0787)
    METEOR toàn bài 0,7774 → 0,7598   (−0,0176)

thí nghiệm T10 (đo lại độc lập, cơ chế khác):
    thêm ~59 âm tiết vào một đáp án HOÀN HẢO  →  mất 0,045–0,05
```

Khối 3 tốt lên, toàn bài tụt. METEOR phạt viết dư. `configs/models.json` giữ 256/160 là
**quyết định có căn cứ**, không phải bỏ sót.

**Một hướng KHÔNG đóng vĩnh viễn:** phép đo về dẫn chiếu thực hiện trên rổ có 47,2% sai
Điều. Nếu chất lượng rổ tăng đáng kể thì đó là bài toán khác.

---

## 7. Giới hạn của nghiên cứu này

Liệt kê đầy đủ, kể cả những giới hạn chưa khắc phục được.

### 7.1 Cấu hình đã nộp được chọn trên chính tập dùng để chấm nó

Đây là giới hạn nghiêm trọng nhất, và nó áp lên **con số duy nhất sống sót** của cả đợt
thí nghiệm: hiệu số +0,0337 ở §4.4 được đo trên đúng 1.000 câu đã dùng để quét cả ba nút.

| Nút | Chọn trên | Thiên lệch do chọn |
|---|---|---|
| α = 0,7 | chính 1.000 câu này | **đã định lượng ≈ 0,002** — bootstrap toàn mẫu +0,0082 so kiểm chéo hai nửa +0,006 |
| k = 2 (số đoạn giao) | chính 1.000 câu này | **chưa kiểm chéo** |
| ngân sách 200 âm tiết | chính 1.000 câu này | **chưa kiểm chéo** |

Một nút đã đo được thiên lệch, hai nút chưa. Nếu cả ba nút chịu thiên lệch cùng cỡ với α
thì phần lạc quan tổng cộng vào khoảng 0,006 — nhưng đó là **phép ngoại suy, không phải
phép đo**, và nó giả định ba nút độc lập, điều chưa được kiểm.

Hai điều làm giới hạn này nhẹ bớt, nhưng không xoá nó:

- Hiệu ứng của `k=3 → k=2` lớn và có cơ chế giải thích được (khi hạng 1 đúng ~65% thì
  đoạn thứ ba chỉ pha loãng; khi hạng 1 chỉ đúng ~39% thì nó là cứu cánh). Hiệu ứng lớn
  kèm cơ chế ít có khả năng là hiện vật của việc chọn.
- Con số **0,5383 trên bảng private là số sạch** — tập đó chưa từng được dùng để chỉnh
  bất cứ nút nào.

Cách khắc phục đã biết nhưng không kịp thực hiện: quét lại cả ba nút với kiểm chéo, trên
mẫu ≥830 câu (§2.8), dùng 6.000 câu train chưa đụng tới.

**Tập đánh giá của LegalQA đã bị lọc sẵn.** 1.000 câu dùng để đánh giá tách được ba khối
**100%**, trong khi dân số chung chỉ **53,4%** — vì harness cần tách được Khối 1 và Khối
3 từ đáp án chuẩn mới chạy. Đã đo trực tiếp nhóm không lọc (200 câu mỗi nhóm):

```
nhóm lọc      0,5429        nhóm không lọc  0,5446
chênh lệch  +0,0018   CI95 [−0,0382; +0,0423]
```

**Không có hiệu ứng lớn.** Nhưng CI rộng ±0,04 trong khi khoảng cách cần giải thích là
0,0154 — phép đo **không đủ phân giải** để xác nhận hay loại trừ một hiệu ứng cỡ đó. Câu
nói được là *"bộ lọc không làm tập đánh giá dễ hơn một cách rõ rệt"*; câu **không** nói
được là *"bộ lọc không ảnh hưởng gì"*.

**Khoảng cách 0,0154 giữa đo nội bộ và LB chưa giải thích được.** Đã loại trừ độ khó của
khâu truy hồi: phân bố điểm truy hồi của tập private gần như trùng tập train, và tái trọng
số theo hai trục độc lập (điểm truy hồi top-1; độ dài context) đều giải thích ≈0. Hai khả
năng còn lại — thiên lệch do chọn tham số trên train, và khác biệt phía đáp án chuẩn —
**không tách được**, vì đáp án chuẩn của tập private không quan sát được. Tập giữ riêng
thứ hai (bảng public) đã khoá trước khi đo được.

**Nhãn dữ liệu huấn luyện là nhãn suy ra, không phải nhãn người gán.** Cặp (câu hỏi ↔
Khoản) dựng bằng đối chiếu chuỗi với F1, thu hẹp bằng BM25 top-20. Thiên lệch đã biết:
câu nào BM25 trượt văn bản đúng thì **mất nhãn**, và thiên lệch đó nghiêng về phía "dễ" —
tập dạy thiếu đúng nhóm câu khó cần cải thiện nhất.

**Một số kết quả của giai đoạn đầu không tái lập được.** Script gốc không còn trên đĩa;
chúng được đánh dấu riêng trong nhật ký và **không dùng để ra quyết định**.

**Khuyết tật đã biết, đã định lượng, chọn không vá:** 14/1.918 câu private (0,73%) có một
context vượt ngân sách 200 âm tiết tới 80–325 lần. Giá đo được trên train: **0,0012** —
nhỏ hơn 20 lần hiệu số cấu hình, và khuyết tật này đã có mặt trong chính lượt đo nên con
số 0,5537 đã tính cả nó.

---

## 8. Tái lập

Repo **không chứa dữ liệu** — dữ liệu BTC cấp không được tái phát tán. Đặt dữ liệu gốc
vào `../data/` rồi chạy theo `README.md` của từng nhánh.

Mọi tham số ngẫu nhiên đều cố định: seed 2026 cho sinh câu trả lời, seed 42 cho bootstrap,
giải mã greedy (không dùng RNG). Tên file đầu ra suy từ cấu hình để không thể dán nhầm
nhãn giữa hai bản build khác rổ.

**Kiểm tương đương khi đổi hạ tầng.** Khi chuyển sang sinh theo lô và hai GPU, phép kiểm
là: Khối 1 phải khớp **từng byte** với bản cũ. Kết quả: **40/40 khớp**; 6 câu lệch, và
**cả 6 đều là trường hợp Khối 3 trước rỗng nay có nội dung** — tức mọi chỗ lệch nằm đúng
ở chỗ bản vá nhắm tới. Phép kiểm này mạnh hơn md5 toàn file vì nó chỉ ra **vị trí** của
chỗ lệch.

Cấu hình sinh câu trả lời dùng giải mã greedy nên seed không vào RNG; gom lô do đó không
đổi kết quả về mặt lý thuyết, và phép kiểm trên xác nhận điều đó bằng thực nghiệm.

---

## 9. Nguồn

| Nội dung | Tài liệu |
|---|---|
| Nhật ký đầy đủ theo thời gian | `BAO_CAO_TONG_HOP_THI_NGHIEM.md` |
| LegalIR — nhật ký kỹ thuật, hướng đã đóng | `IR/part2-xep-hang/docs/NHAT_KY_KY_THUAT.md` |
| LegalIR — bảng 10 bộ chấm | `IR/part2-xep-hang/docs/bang_ablation.md` |
| LegalQA — mục lục quyết định, một dòng mỗi thí nghiệm | `QA/part1-truy-xuat/retrieval/docs/NHAT_KY_THI_NGHIEM_RETRIEVAL.md` |
| LegalQA — điều kiện đo và bằng chứng từng thí nghiệm | `QA/part1-truy-xuat/retrieval/docs/KET_QUA_THUC_NGHIEM_18-09.md` |
| LegalQA — generator, các phiên bản baseline | `QA/part2-sinh-cau-tra-loi/reports/` |

Bốn repo mã nguồn: `justySusanto/Task1_Retrieval` · `Loc2116/DSC_2026_Task1_Part2` ·
`maidang09122006-ux/ChanTaooDe-` · `Hakuchoiii2/LegalQA_Task-2`.
