# DSC 2026 — Tổng hợp thí nghiệm hai nội dung thi

**Phạm vi:** mọi thí nghiệm đã chạy trên Task 1 (LegalIR) và Task 2 (LegalQA), cả phần
truy hồi lẫn phần sinh câu trả lời. Gộp từ bốn repo thành viên, hai harness nội bộ và
mười ba tài liệu kết quả.

> **File này là NHẬT KÝ — viết cho người đang làm.** Thứ tự thời gian, sai rồi sửa, mỗi
> mục kèm ngưỡng và điều kiện đo.
>
> Bản viết cho **người phản biện và BTC** — kết luận trước, bằng chứng sau — nằm ở
> [`BAO_CAO_PHUONG_PHAP_VA_KET_QUA.md`](BAO_CAO_PHUONG_PHAP_VA_KET_QUA.md).
>
> **Hai file được giữ tách rời có chủ ý.** Gộp lại sẽ làm mất phần *điều kiện đo* đi kèm
> mỗi con số, mà mất điều kiện đo chính là căn bệnh đã gây ra bốn lỗi ghi trong file này.

**Cập nhật:** 20/09/2026 · **Trạng thái:** Task 1 chốt public LB **0,9124** · Task 2 đã
nộp private, **LB 0,5383**

| Con số chốt | |
|---|---|
| Task 1 — LegalIR, public LB | **0,9124** (recall@5) |
| Task 2 — LegalQA, private LB | **0,5383** (METEOR) |
| Task 2 — dư địa còn lại ở khâu xếp hạng | trần trong rổ hiện có **+0,1159** (thang harness — xem 3.9) |

---

## 0. Đọc file này thế nào

Bốn quy ước, vi phạm một cái là đọc sai cả bảng.

**Không có thang đo chung.** Task 1 chấm bằng recall@5 trên 5 mã văn bản; Task 2 chấm
bằng METEOR trên văn bản sinh ra. Trong Task 2 lại có ít nhất năm harness khác nhau
từng cho ra các mốc 0,42 / 0,49 / 0,63 / 0,77 / 0,83 — tất cả đều là "METEOR" nhưng đo
những thứ khác nhau. **Mỗi con số trong file này đi kèm điều kiện đo của nó.** Đừng bốc
một số ra khỏi hàng của nó.

**Ba mức tin cậy của nguồn:**

| Ký hiệu | Nghĩa |
|---|---|
| `đo` | script sinh ra con số này còn trên đĩa, chạy lại được |
| `ghi nhận` | có trong báo cáo, script gốc không còn — đừng trích như số đã kiểm |
| `ước` | suy ra từ mô hình hoặc phép ngoại suy, không phải phép đo |

**Thí nghiệm thất bại được giữ nguyên.** Phần 2.5 và 3.5 liệt kê các hướng đã đóng.
Mỗi dòng là một phép đo đã trả tiền bằng giờ GPU; biết hướng nào không đi được đáng
đúng bằng biết hướng nào đi được.

**Δ luôn kèm cách đo.** Chênh lệch giữa hai cấu hình chỉ có nghĩa khi ghép cặp trên
cùng tập câu. Chỗ nào có khoảng tin cậy, nó được bootstrap ghép cặp 2.000 lượt seed 42.

---

## 1. Hai bài toán, một bộ dữ liệu

```
kho 8.532 văn bản pháp luật  (selected-contexts, md5 dccc8257f2fc4da9988d3687a56980e2)
           │
    ┌──────┴──────┐
    ▼             ▼
 TASK 1        TASK 2
 LegalIR       LegalQA
 trả về 5      trả về đoạn văn trả lời câu hỏi
 mã văn bản    theo cấu trúc ba khối
 recall@5      METEOR
```

Task 2 chia câu trả lời làm ba khối, và **tỷ trọng điểm rất lệch** — số này chi phối
toàn bộ ưu tiên của Task 2:

| Khối | Nội dung | Ai sinh | Điểm mất nếu bỏ hẳn |
|---|---|---|---:|
| Khối 2 | trích nguyên văn điều luật | **chương trình chép** từ context | **0,5868** |
| Khối 3 | kết luận | model | 0,3129 |
| Khối 1 | câu dẫn căn cứ | model | 0,0841 |

`đo` trên 479 câu tách được đủ ba khối từ `qa_packages_heldout_2026-08-31.json`.

Khối 2 — thứ nặng nhất — **không do model sinh ra**. Nó là đầu ra của khâu truy hồi.
Đó là lý do phần lớn công sức Task 2 đổ vào truy hồi chứ không vào model.

---

## 2. TASK 1 — LegalIR

Hai chặng, hai người: Part 1 sinh ứng viên (thành viên D), Part 2 xếp hạng và chốt
top-5 (thành viên E).

**Bài nộp cuối:** `submission_matchEmb_K50_n2.zip` — **public LB 0.9124** (30/08).
Cấu hình: rổ `_matchEmbedded` · `M=20` · `K=50` · `MERGE_CHARS=1800` · `blend n_bm25=2` ·
reranker `AITeamVN/Vietnamese_Reranker`.

### 2.1 Part 1 — sinh ứng viên

| ngày | việc | tập đo | kết quả |
|---|---|---|---|
| 08/08 | chunking + BM25 baseline | dev 150 | trần 0,9933 · R@100 0,9700 · R@5 0,7578 |
| 13/08 | đổi sang dev lồng nhau 1000/300 | dev 300 | trần 1,0000 · R@100 0,9783 · R@5 0,7533 |
| 17/08 | bake-off 3 bi-encoder + BM25 | dev 1000 | trần 0,9980 · BM25 R@50 0,9424 |
| 20/08 | hoà RRF, chốt artefact top-50 | dev 1000 | **RRF R@50 0,9738** |

> Hàng 08/08 đo trên tập dev 150 câu **đã khai tử** — 256/300 câu của nó về sau lọt vào
> tập huấn luyện. Không so được với ba hàng dưới.

**Bake-off bi-encoder** (17/08, dev 300):

| model | R@100 | chồng lấp với BM25 | thời gian |
|---|---:|---:|---:|
| `BAAI/bge-m3` | **0,9917** | 37,18% | ~6,2 giờ |
| `Qwen/Qwen3-Embedding-0.6B` | 0,9817 | 36,40% | ~3,9 giờ |
| `bkai/vietnamese-bi-encoder` | 0,9467 | 31,75% | ~1,2 giờ |

**Kết quả chính của Part 1:** chồng lấp giữa nhánh từ vựng và nhánh ngữ nghĩa chỉ
31–37%, nghĩa là hai nhánh bù trừ nhau tốt. Hoà RRF ăn đúng phần bù trừ đó:

```
BM25 đơn thuần      R@50 = 0,9424
RRF (BM25 ⊕ dense)  R@50 = 0,9738      +3,14 điểm
trần của chunking          0,9980      RRF còn cách trần 2,42 điểm
```

### 2.2 Part 2 — xếp hạng

**Chuỗi tăng tiến trên public LB.** Mọi Δ đo so với mốc `AITeamVN` gốc + `blend n=2`:

| bước | public LB | Δ |
|---|---:|---:|
| rerank AITeamVN, n=0 | 0.8350 | — |
| + `blend_bm25_first(n=2)` | 0.8573 | +2,23 |
| + deepchunk M=20/K=20, `max` | 0.8840 | +2,67 |
| + BM25 chọn đoạn + gộp 1800 ký tự | 0.8871 | +0,31 |
| + rổ FUSION (M=10, n=3) | 0.9063 | +1,92 |
| + M=20, n=2 | 0.9109 | +0,46 |
| + rổ `_matchEmbedded`, n=3 | 0.9118 | +0,09 |
| + `K=50`, n=2 | **0.9124** | +0,06 |

**Đọc bảng:** hai đòn lớn nhất — đổi reranker (+11 trên dev) và đổi rổ ứng viên (+2,4)
— đều là **đổi thứ nạp vào**, không phải tinh chỉnh cái đang có.

**Bảng 2×2 `K` × `n`** (30/08), hai cần gạt cuối cùng còn mở:

| public LB | **n=2** | **n=3** | hiệu ứng `K` |
|---|---:|---:|---:|
| **K=20** | 0.9104 | 0.9118 | |
| **K=50** | **0.9124** ← chốt | 0.9118 | |
| hiệu ứng `n` | +0,20 | 0,00 | |

`K` và `n` **thay thế nhau** — cả hai nhắm vào cùng chỗ bộ chấm làm hỏng. `n` che bằng
cách giữ chỗ cho rổ; `K=50` chữa bằng cách cho bộ chấm đọc kỹ hơn. Làm một cái rồi thì
cái kia hết việc: `K` ăn +0,20 ở `n=2` nhưng đúng **0,00** ở `n=3`.

> **Giá thật của hướng `K=50`: 14,8 giờ GPU cho +0,06 điểm** (0,6 câu trên 1000).

### 2.3 Bảng đối chứng 10 bộ chấm chéo

Giao thức: 300 câu dev khoá cứng · rổ RRF top-50 · **15.000 cặp giống hệt nhau cho mỗi
model** · chấm một tầng, không đọc sâu · T4. Trần của rổ (recall@50) = 0,9817.

| Model | Nền | Tham số | R@1 | R@5 | MRR@10 | giây/cặp |
|---|---|---:|---:|---:|---:|---:|
| **AITeamVN/Vietnamese_Reranker** ← dùng | XLM-R, tinh chỉnh tiếng Việt | 0,568B | **0,5017** | 0,7467 | **0,6178** | 0,058 |
| BAAI/bge-reranker-v2-m3 | XLM-R đa ngữ | 0,568B | 0,4683 | **0,7517** | 0,5994 | 0,056 |
| jina-reranker-v2-base-multilingual | XLM-R đa ngữ | 0,278B | 0,4383 | 0,7233 | 0,5677 | 0,025 |
| mmarco-mMiniLMv2-L12 | mMiniLM đa ngữ | 0,118B | 0,4867 | 0,7217 | 0,5963 | **0,007** |
| Qwen3-Reranker-0.6B | Qwen3 **decoder** | 0,596B | 0,4133 | 0,7217 | 0,5627 | 0,257 |
| itdainb/PhoRanker | PhoBERT (trần 256 token) | 0,135B | 0,4283 | 0,7200 | 0,5549 | 0,014 |
| namdp-ptit/ViRanker | XLM-R tiếng Việt | 0,568B | 0,3917 | 0,6867 | 0,5182 | 0,106 |
| BAAI/bge-reranker-base | XLM-R base | 0,278B | 0,3583 | 0,6850 | 0,5064 | 0,018 |
| ms-marco-MiniLM-L6-v2 *(đối chứng ÂM)* | BERT tiếng Anh | 0,023B | 0,3233 | 0,5633 | 0,4421 | 0,005 |

Bốn điều bảng này chứng minh:

1. **Model đang dùng là lựa chọn đúng** — dẫn đầu ở R@1 và MRR@10, đúng hai chỉ số
   pipeline phụ thuộc. Nó chỉ thua nền của chính nó (`bge-v2-m3`) ở R@5 thuần: phần
   tinh chỉnh tiếng Việt đổi một chút R@5 lấy khả năng xếp đúng ở đỉnh.
2. **`thanhtantran/Vietnamese_Reranker` là bản sao** — điểm trùng khít từng bit với
   `AITeamVN` trên cả 15.000 cặp. Khác tài khoản HF, cùng trọng số. Đã loại khỏi bảng.
3. **Số tham số không dự đoán được chất lượng** — `mMiniLMv2` 0,118B hơn `ViRanker`
   0,568B tới +3,50 điểm R@5, dù nhỏ hơn 4,8 lần và nhanh hơn 16 lần. Thứ quyết định
   là dữ liệu huấn luyện tiếng Việt, không phải kích thước.
4. **Kiến trúc decoder không tự động tốt hơn** — `Qwen3-Reranker-0.6B` đạt đúng 0,7217,
   **bằng** `mMiniLMv2` 0,118B, nhưng chậm hơn 40 lần.

Đối chứng âm hoạt động đúng: model tiếng Anh thuần rơi 18,8 điểm dưới nhóm dẫn đầu.

### 2.4 Phát hiện mạnh nhất của Task 1

**Reranker teo lại khi rổ mạnh lên.**

| dev300 | recall@5 |
|---|---:|
| rổ trần @5 — **không chấm lại gì** | **0,9117** |
| tầng 1 đầy đủ (k=3, MAX) | 0,9100 ← *kém hơn không chấm* |
| `blend n=3` (cấu hình nộp) | 0,9467 |
| trần tuyệt đối (recall@50 của rổ) | 0,9883 |

Chấm lại 50 ứng viên bằng cross-encoder 568 triệu tham số ra **kém hơn không chấm gì**.
Reranker không yếu — rổ đã tốt tới mức không còn gì để sửa.

13 câu còn sai: 3 ngoài rổ · 1 gold ở hạng 5 bị CE đẩy ra · 9 cả rổ lẫn CE đều vùi.

**Vì sao thêm tham số không cứu được nhóm này** (đọc thật 2 trong 13 ca): model tối ưu
độ khớp câu chữ, còn đáp án đúng đòi biết **văn bản nào có thẩm quyền**. Ca `138214`:
model chọn Quyết định UBND Hà Nội thay vì Nghị định toàn quốc — trùng chữ nhiều hơn
nhưng sai phạm vi. Cross-encoder chấm tương đồng câu hỏi–đoạn **không có chỗ biểu diễn
thứ bậc pháp lý**. *(Mới đọc 2/13 ca — giả thuyết có ví dụ, chưa phải số liệu.)*

### 2.5 Task 1 — hướng đã đóng

| # | hướng | kết quả |
|---|---|---|
| 1 | fine-tune toàn phần (2 lần) | **−0,50** rồi **−6,50** |
| 2 | chưng cất từ Qwen3-8B | NET −16 nhóm/600. Thầy 0,5417 **thua** trò 0,5683 |
| 3 | probe — học lại đầu phân loại | Δ tốt nhất +0,67; **18 câu sai: 5/18 → 5/18, không đổi câu nào** |
| 4 | `M=30` | trần ~0 (3 câu chạm tới) |
| 5 | ensemble nhiều bộ chấm | âm |
| 6 | trộn hai pipeline (rổ cũ ⊕ rổ mới) | âm |
| 7 | gộp điểm z-score / RRF hai thứ hạng | **thua `blend n=3` ở MỌI tham số** |
| 8 | chọn `n` thích nghi theo câu | trần oracle +1,17 — dưới ngưỡng |
| 9 | ưu thế theo loại văn bản / thẩm quyền | âm, kể cả khi ước tham số ngay trên dev300 |
| 10 | ưu tiên văn bản mới / còn hiệu lực | gold là văn bản mới nhất đúng **2,0%** = ngẫu nhiên |
| 11 | ghép tên văn bản | slug không dấu −2,0; tiêu đề có dấu +1,00 nhưng **83% trùng** tầng 2 |
| 12 | LLM ngoài (Gemini/GPT) | hoà, **và BTC cấm** |
| 13 | "gợi ý liên quan" kiểu Google | thua cả ba biến thể |
| 14 | tăng `top_chunks` 3→5 | đã rút lại |
| 15 | cắt ứng viên 100→50→25 | đòn chi phí, không phải đòn điểm |
| 16 | cải tiến `pick_chunks` | `K=20` đã bắt 90% giá trị oracle |

### 2.6 Bảng quan trọng nhất của Task 1 — độ lệch dev ↔ LB

| ngày | cấu hình | dev300 | public LB |
|---|---|---:|---:|
| 12/08 | candidate D, n=2 | 0.8883 | 0.8573 |
| 22/08 | rổ fusion M=10, **n=1** | **0.9350** ← đỉnh dev | 0.8943 ← **thấp nhất LB** |
| 22/08 | rổ fusion M=10, n=2 | 0.9283 | 0.9033 |
| 22/08 | rổ fusion M=10, **n=3** | 0.9233 ← thấp nhất dev | **0.9063** ← đỉnh LB |

**Không có hằng số quy đổi.** Độ lệch thay đổi theo `n` (−5,08 / −4,07 / −2,50), và dev
đã từng **đảo dấu**: chênh dev −1,17 ứng với chênh LB **+1,20**. Rổ `_matchEmbedded` là
ví dụ đắt nhất: **dev +3,83 → public +0,11**.

> dev300 dự báo được **thứ tự** khi hiệu ứng lớn, **không** dự báo được mức tuyệt đối.

**Nhưng dev300 mạnh hơn tưởng ~3 lần nếu so đúng cách.** Sai số không ghép cặp là 1,30
điểm, nhưng mọi phép so trong dự án đều ghép cặp — cùng 300 câu, đổi một tham số — nên
chỉ câu **bất đồng** mới mang thông tin. McNemar:

| cặp | A thắng | B thắng | bất đồng | Δ | p |
|---|---:|---:|---:|---:|---:|
| n=2 vs **n=3** | 1 | 6 | 7 | +1,67 | 0,125 |
| n=1 vs **n=2** | 1 | 8 | 9 | +2,33 | **0,039** |
| n=0 vs **n=3** | 2 | 23 | 25 | +7,00 | **<0,0001** |

Chi phí: 0 giờ GPU.

---

## 3. TASK 2 — LegalQA

Ba người: B (truy hồi), C (sinh câu trả lời), và phần đóng gói context.

**Bài nộp tốt nhất hiện tại:** V5 (đóng gói) + V8.3 Full Hard (model), **LB 0,4724**.

### 3.1 Phân rã mất điểm — bảng định hướng mọi thứ còn lại

Phương pháp: thay model thật bằng **generator hoàn hảo** — lấy Khối 1 và Khối 3 thẳng
từ đáp án chuẩn, chỉ đổi Khối 2. Mọi chênh lệch còn lại đều do context. Ghép lại ba
khối gốc cho đúng METEOR = 1,0000 nên phép tách không mất chữ.

| Khối 2 dùng gì | METEOR |
|---|---:|
| oracle context | 1,0000 |
| truy xuất thật, top-3 | 0,6651 |
| truy xuất thật, top-1 | 0,6038 |
| bỏ hẳn Khối 2 | 0,4132 |

| Nguồn mất điểm | Mất |
|---|---:|
| generator (oracle 1,0 → V8.3 0,8289) | 0,171 |
| **truy xuất** (generator hoàn hảo 1,0 → 0,6651) | **0,335** |
| `ước` dự đoán LB | 0,494 |
| LB thật của V1 | 0,4506 |

Sai số 0,043 — hai nguồn giải thích gần hết khoảng cách. **Truy xuất tốn gấp đôi
generator.** Đó là căn cứ để dồn công vào truy hồi.

Truy xuất trúng ở mức nào (top-3): đúng văn bản 76,5% · đúng Điều 51,4% · **đúng Khoản
39,5%**. Con số 76,5% khớp độc lập với P@1 = 0,7300 mà nhóm truy xuất đo trên dev300.

### 3.2 Phần truy hồi — bảng thí nghiệm

Đường ống bốn chặng: BM25 cấp văn bản → nở ra Khoản → bi-encoder trên 432.473 Khoản →
cross-encoder → đóng gói.

| Mã | Câu hỏi | Ngưỡng đặt trước | Kết quả | Quyết định |
|---|---|---|---|---|
| **E1** | Trùng `unit_id` có hại không? | trùng xuyên văn bản > 1% | 7.631 id trùng, **100% cùng một Điều**, 0 xuyên văn bản | không chạm → xuống cuối hàng đợi |
| **E3a** | Nở đoạn ra cả Điều? | phủ chữ tăng ≥ 10 điểm | 63,5%→82,5% (rổ thật k=3, n=472)<br>73,9%→91,5% (BM25 top-20, n=300) | chạm, nhưng E6 bác |
| **E6** | Ngân sách âm tiết? | tìm cực trị | METEOR đỉnh ở 300; nở cả Điều **thấp hơn** | **phủ chữ là chỉ số sai** |
| **E6b** | 3 bộ chọn × ngân sách | đỉnh lệch 350 quá ±100 | đỉnh 300/500/400, phẳng 300–500 | giữ 350 (sau đổi — xem 3.4) |
| **E5** | Đọc dẫn chiếu sang Điều khác? | METEOR ≥ +2,0 điểm | +0,0121; **91,9% dẫn chiếu trỏ trong cùng văn bản** | không chạm → loại bản top-50 |
| **E2** | Vá mã trùng | bật `dedup_by_row` phải hơn nhiễu | rổ mất 0,58–1,04 suất/câu, lấy lại chỉ METEOR +0,0003/+0,0014, CI sát 0 | **giữ bản vá, không bật cờ** |
| **E4** | Nhúng kèm tiêu đề Điều? | ≥ +2,0 **và** cận dưới CI > 0 | phủ ≥80%: 76,4%→**80,8%** (+4,40, CI [+3,00,+5,90]); METEOR +0,0217 | **ĐẠT — thí nghiệm đầu tiên vượt ngưỡng** |
| **T1** | Gold ở hạng mấy trong 50? | ngoài 50 ≥ 50% → sửa nguồn ứng viên | ngoài 50 chỉ **9,6% / 13,7%** | nghẽn ở **xếp hạng**, không phải sinh ứng viên |
| **T2** | Lỗ cắt 50.000 ký tự | — (đo phơi nhiễm) | BM25 mù **25,39%** kho; 21,89% gold sau chỗ cắt | **không vá** — xem 3.3 |
| **T5** | Vì sao 75 câu tệ đi sau E4? | — (kiểm giả thuyết) | giả thuyết **bị số liệu bác**, cả hai biến ngược chiều | đóng |
| **T3** | Trộn hai bộ xếp hạng | — | đỉnh **ở giữa**, cao hơn cả hai đầu | **ĐẠT** |
| **T6** | Xếp hạng hoàn hảo trong rổ 50 thì được bao nhiêu? | — (đo trần) | +0,1159 so sản xuất; **71% dư địa nằm trong top-10** | trần lớn, nhưng xem T9 trước khi diễn giải — 3.9 |
| **T9** | Chấm lại chỉ trong cửa sổ top-W | ≥+0,010 **và** CI>0 **và** thắng nửa giữ riêng | +0,0019 / CI [−0,0003,+0,0044] / kiểm chéo **−0,0002** | **TRƯỢT CẢ BA → ĐÓNG.** α đã làm sẵn việc của W — xem mục 7 |
| **T10** | Viết Khối 3 khi gold không có thì mất bao nhiêu? | — | nhóm cần đo −0,0508, **nhóm đối chứng −0,0449** → phần riêng chỉ −0,0059 | **ĐÓNG** — xem 6.8 |
| **T11** | Tập đánh giá bị lọc sẵn có làm nó dễ hơn không? | — | lọc 0,5429 · không lọc **0,5446** · Δ +0,0018 CI [−0,0382, +0,0423] | không có hiệu ứng **lớn**; CI quá rộng để kết luận về 0,0154 — xem 6.9 |
| **P2** | Fine-tune bi-encoder mức Khoản | recall@50 ≥ +2,0 **và** CI>0 | recall@50 **−0,23 điểm**, CI [−2,96, +2,51] | **TRƯỢT → ĐÓNG** — xem mục 7 |
| — | Dựng dữ liệu dạy mức Khoản cho P2 | — | 2.597 cặp / 6.000 câu (43,3%); **phát hiện tập đánh giá bị lọc sẵn** | xem 6.7 |
| **T7** | Khoảng cách train→private do đâu? | — (phân rã) | truy hồi giải thích **≈0** của −0,0154 | xem 6.4 |
| **T8** | Chuẩn hoá Unicode NFC bài nộp | — | **−0,0004**, CI loại trừ 0 | **không chuẩn hoá** — xem 6.5 |

> **Hai con số E3a KHÔNG so được với nhau.** E3a-heldout khởi đầu ở 63,5%, E3a-bm25 ở
> 73,9% — vì 20 đoạn đương nhiên che nhiều chữ hơn 3 đoạn. Nên +19,0 và +17,6 **không
> phải** hai lần đo cùng một hiệu ứng ra gần nhau. Đừng lấy trung bình.

### 3.3 Bốn phát hiện đáng kể nhất của phần truy hồi

**(a) Phủ chữ là chỉ số sai để chọn tham số.** E3a cho thấy nở đoạn ra cả Điều che
được nhiều chữ gold hơn hẳn (63,5% → 82,5%). Nhưng chấm bằng METEOR — thước thật của
cuộc thi — điểm lại **tụt**. Lý do: phủ chữ chỉ đếm phần đúng, không trừ phần dư, nên
nó **đơn điệu tăng** theo ngân sách và không bao giờ có cực trị. METEOR có hình phạt
phân mảnh nên có cực trị. Từ đó mọi lựa chọn tham số chấm bằng METEOR.

**(b) Bỏ RRF đã vô hiệu hoá nhánh BM25 — tác dụng phụ không ai ghi lại.**

`search_units_hybrid` = union(BM25→nở, dense top-300) → **xếp bằng điểm dense** → cắt 50.

Vì `dense top-300 ⊇ dense top-50` và xếp thuần bằng điểm dense, một unit do BM25 thêm
vào chỉ lọt top-50 nếu điểm dense của nó vượt unit thứ 50 — mà nếu vậy nó đã nằm sẵn
trong dense top-50. **Về mặt cấu trúc, BM25 không thể đóng góp gì.** Đo được:

```
trùng lặp rổ-cuối ↔ dense top-50   96,8% (bare) · 96,2% (title)
unit KHÔNG có trong dense top-50   1,6/50 · 1,9/50
số câu rổ-cuối TRÙNG HOÀN TOÀN     43,8% · 45,9%
```

Quyết định xếp thuần bằng dense (Hit Khoản@3 55,5% so với 27,0%) đúng ở thời điểm đó,
nhưng nó giết nhánh BM25 như một tác dụng phụ, và điều đó **sống sáu tuần không ai
biết**. Hệ quả: lỗ cắt 50.000 ký tự ở T2 — dù BM25 mù một phần tư kho — **không chảy
vào điểm**. Không vá.

**(c) Reranker có trần, nâng nhánh yếu và dìm nhánh mạnh.**

```
nhánh bare  (yếu)   54,9%  →  60,1%     reranker  +5,2
nhánh title (mạnh)  64,8%  →  60,7%     reranker  −4,1
```

Cả hai rơi về cùng một chỗ — dấu hiệu reranker **ghi đè** thứ tự nền bằng ý kiến riêng,
và ý kiến riêng của nó đáng đúng khoảng 60%. Một bộ xếp hạng lại chỉ sửa **thứ tự**,
không sửa **sự có mặt**: vô dụng khi đáp án vắng mặt trong rổ, có hại khi thứ tự nền đã
tốt hơn nó. Cách chữa không phải gỡ nó, mà cho nó **một phần** tiếng nói.

**(d) Xếp hạng tốt lên thì số đoạn tối ưu giảm xuống.** Đo trên harness cả bài (Khối 1
và 3 lấy từ gold, chỉ thay Khối 2), n=1.000:

| ngân sách | k=1 | k=2 | k=3 | dài thật ở k=2 |
|---:|---:|---:|---:|---:|
| 150 | 0,6714 | 0,7119 | 0,6935 | 376 âm tiết |
| **200** | 0,6848 | **0,7135** ← đỉnh | 0,6865 | 450 |
| 250 | 0,6935 | 0,7120 | 0,6784 | 525 |
| 350 | 0,7005 | 0,7066 | 0,6648 | 599 |
| 500 | 0,7030 | 0,6983 | 0,6521 | 676 |

Quy về **tổng độ dài giao đi** thì cả ba cột cùng chỉ về một chỗ: tối ưu quanh 450–500
âm tiết, còn chia mấy đoạn là chuyện phụ. Gold Khối 2 median ~200 âm tiết.

Phép so một biến (cùng ngân sách, cùng α, cùng nhánh):

```
ngân sách 350:  k=3 0,6648 → k=2 0,7066   +0,0418  CI95 [+0,0365, +0,0466]
ngân sách 200:  k=3 0,6865 → k=2 0,7135   +0,0271  CI95 [+0,0216, +0,0320]
```

Kết quả này **ngược** một phép đo cũ (ghép 3 đoạn 0,6651 > ép 1 đoạn 0,6515). Giải
thích: phép đo cũ chạy trên rổ có hạng 1 đúng ~39%; rổ hiện tại đúng 65%. Khi hạng 1
thường đúng thì đoạn thêm vào chỉ pha loãng; khi hạng 1 hay sai thì đoạn thêm vào là
cứu cánh. **Đổi một tham số làm tham số khác đổi nghĩa.**

### 3.4 Cấu hình truy hồi chốt cho vòng private

| Nút | Trước (V8) | Sau (V9) | Căn cứ |
|---|---|---|---|
| Nhúng cái gì | Khoản trần | **+ tiêu đề Điều** | E4, T1 |
| Xếp thứ tự | reranker toàn quyền | **RRF trọng số α = 0,7** | quét α + kiểm chéo hai nửa |
| Số đoạn giao | 3 | **2** | bảng 3.3(d) |
| Ngân sách mỗi đoạn | 350 | **200** | bảng 3.3(d) |
| Giới hạn tiêu đề | không | **không** | F2 đo không có giới hạn |

Công thức trộn — nghịch đảo hạng chứ không trung bình tuyến tính, vì sản xuất chỉ giao
2 đoạn nên chỉ đỉnh danh sách có giá trị:

```
s = α/(60 + dense_rank) + (1−α)/(60 + rerank_rank)
```

**Về α = 0,7, nói cho đúng:** quét trên mẫu train seed 1909 cho đỉnh **phẳng** trong
dải 0,6–0,8. Kiểm chéo hai nửa chọn 0,6 và 0,8, cả hai đều thắng α=1,0 trên nửa giữ
riêng, không đảo dấu. Lấy trung điểm. Lợi ích **ước bằng kiểm chéo là +0,006**, không
phải +0,0082 của bootstrap toàn mẫu — chênh lệch giữa hai con số chính là độ thiên lệch
do chọn bằng mắt. Câu nói được là *"α không thua dense-only, và thắng rõ rerank-only"*.

### 3.5 Phần sinh câu trả lời — baseline V1 đến V7

Mọi số dưới đây đo với **oracle context** trên 100 câu validation, seed 2026. **Không
so sánh được với LB.**

| Bản | Thay đổi chính | Kết quả |
|---|---|---|
| **V2** | tách độc lập `<LEAD>`/`<CONCLUSION>`, retry một lần, fallback template Khối 1 | Qwen3-1.7B 0,7808 · đủ khối sau retry 95% |
| **V3** | targeted retry — chỉ sinh lại khối đang thiếu | đủ ba khối **100%** cả ba model; phục hồi Khối 3 100% |
| **V4** | V3 nhưng giữ nguyên văn prompt V2 (ablation sạch) | Qwen3-1.7B **0,7774** ← *tốt nhất cả nhóm baseline* |
| **V5** | bỏ giới hạn 120 từ ở Khối 3 | toàn bài **giảm** 0,7598; nhưng METEOR riêng Khối 3 **tăng** 0,3965→0,4752 |
| **V6** | Khối 3 theo trọng tâm câu hỏi, tập mệnh đề nhỏ nhất | Qwen2.5-3B 0,7753 (ROUGE cao nhất 0,8140); Qwen3-1.7B **giảm** 0,7543 |
| **V7** | V4 + retrieved 2-shot | toàn bài giảm 0,7587; **METEOR Khối 1 tăng vọt** 0,3651→0,4963 |

**Bốn điều nhóm baseline dạy được:**

1. **Ablation sạch là bắt buộc.** V3 đổi cả prompt lẫn cơ chế retry nên Δ không đọc
   được. V4 giữ nguyên văn prompt V2, xác minh raw output lượt đầu khớp **300/300** và
   prediction của mẫu không retry khớp **205/205** — chỉ khi đó Δ mới quy được về
   targeted retry.
2. **Tăng recall một khối có thể làm giảm điểm toàn bài.** V5 là ví dụ sạch nhất: Khối
   3 tốt lên +0,079 nhưng toàn bài mất 0,0176, vì kết luận dài hơn làm tăng recall và
   giảm precision. METEOR phạt viết dư.
3. **Cùng một prompt, ba model phản ứng khác nhau.** V6 thành công với Qwen2.5-3B
   (+0,0105) và Qwen3-0.6B (+0,0168) nhưng thất bại với Qwen3-1.7B (−0,0231). Nguyên
   nhân: Qwen3-1.7B không thực hiện chính sách độ dài ổn định — nhóm đáp án chuẩn ≤40
   từ vẫn sinh 85,2 từ, nhóm >120 từ chỉ sinh 64,0 từ. **Đừng áp một prompt cho mọi
   model rồi giả định phản ứng giống nhau.**
4. **Hai cấu hình tốt nhất thắng ở hai nhóm câu khác nhau.** V4-Qwen3-1.7B thắng 56
   câu, V6-Qwen2.5-3B thắng 44 câu; bộ chọn hoàn hảo sẽ đạt ~0,7992. Router thử nghiệm
   đạt 0,7826 — **nhưng rút ra và chấm trên cùng một tập, nên có nguy cơ overfit và
   chưa được kiểm trên split độc lập.** Chưa dùng.

### 3.6 Phần sinh câu trả lời — V8 QLoRA

| Bản | Thay đổi | METEOR | ROUGE-L |
|---|---|---:|---:|
| V1–V7 | baseline zero-shot / few-shot, không train | tốt nhất 0,7774 | — |
| V8 | **QLoRA adapter** | 0,8139 | 0,8636 |
| V8.1 | greedy + prompt theo loại câu hỏi + quality retry | 0,8257 | 0,8596 |
| V8.2A+B | acceptance gate A₂ + parser nghiêm ngặt B₂ | 0,8267 | 0,8607 |
| **V8.3 Full Hard** | + classifier C + adaptive retry | **0,8289** | 0,8630 |

Base `Qwen/Qwen3-1.7B`, adapter QLoRA, fp16 (**không** lượng tử hoá 4-bit khi chạy),
greedy, seed 2026. Đóng băng từ 31/08.

> **Cảnh báo phải đọc kèm bảng này.** `ước` sai số chuẩn của trung bình METEOR trên 100
> câu là **0,0137**. Các mức tăng V8→V8.1 (+0,0118), V8.1→V8.2 (+0,0010), V8.2→V8.3
> (+0,0022) đều **nhỏ hơn một sai số chuẩn**. Chúng đo trên cùng 100 câu với cùng raw
> output nên phép đo chính xác, nhưng **chưa có bằng chứng chúng còn sống trên câu
> mới**. Thêm nữa, 0,8289 đo trên đúng 100 câu đã dùng để chọn module suốt V8.1→V8.3.
>
> Đòn thật trong chuỗi này là **V7→V8: +0,0365** (fine-tune). Ba bước sau là tinh chỉnh
> dưới ngưỡng nhiễu.

### 3.7 Đóng gói context — bảng version (đo trên LB thật)

| Bản | Cấu hình | LB |
|---|---|---:|
| V1 | top-3, Khoản thô | 0,4506 |
| V2 | top-1, mở rộng cả Điều | 0,4430 |
| V3 | top-1, Khoản thô | 0,3535 |
| **V5** | top-3, ngân sách 300 âm tiết/context, dedup chuỗi y hệt | **0,4724** |
| V6 | thích ứng 1–6 context, mục tiêu 535 âm tiết, dedup Jaccard >0,7 | chưa nộp |

Đây là **bảng duy nhất trong Task 2 đo trên LB thật**. Nó nói hai điều: top-3 hơn top-1
(V1 > V3, +0,0971), và mở rộng cả Điều không bù được (V2 < V1) — cùng chiều với kết
luận E6 về sau.

### 3.8 Task 2 — hướng đã đóng

| Hướng | Vì sao đóng |
|---|---|
| Nở đoạn ra cả Điều | METEOR tụt, dù phủ chữ tăng 19 điểm |
| Phủ chữ làm chỉ số chọn tham số | đơn điệu tăng, không có cực trị |
| Đọc dẫn chiếu từ top-50 | +0,0121 dưới ngưỡng; thêm ~17 Điều rác mỗi câu |
| Vá lỗ cắt 50.000 ký tự | chỉ ảnh hưởng BM25, mà BM25 đóng góp 3% vào rổ cuối |
| Sửa trùng `unit_id` | 100% bản sao cùng một Điều; phơi nhiễm thực tế 1,5% |
| Tách từ bigram cho BM25 | không áp dụng — đã dùng `underthesea.word_tokenize` |
| Cap độ dài tiêu đề | F2 đo không có cap; 4,29% vượt 30 âm tiết |
| Few-shot toàn cục (V7) | METEOR giảm 0,0187; chỉ nhóm trình tự/danh sách tăng |
| Bỏ giới hạn độ dài Khối 3 toàn cục (V5) | toàn bài giảm dù Khối 3 tốt lên |

> Ghi kèm điều kiện: **E5 đo trên rổ có 47,2% sai Điều.** Nếu rổ tốt lên đáng kể thì
> hướng đó là bài toán khác, không nên coi là đóng vĩnh viễn.

### 3.9 T6 — trần của khâu xếp hạng. **Phép đo mở đường cho đợt sau.**

Đo sau khi nộp private, theo quy tắc 9 của Task 1 (*đo trần trước khi chạy*). Cùng rổ
ứng viên, cùng ngân sách, cùng số đoạn — **chỉ đổi thứ tự**. Chọn "tốt nhất" bằng tham
lam theo METEOR của cả bài. Script: `pipeline/eval_tran_xep_hang.py`.

| Mốc (harness cả bài, n=1.000) | METEOR | So sản xuất |
|---|---:|---|
| sản xuất — α=0,7 · k=2 · 200 | 0,7136 | — |
| trần: chọn 2 tốt nhất trong **top-10** | 0,7958 | **+0,0822** CI95 [+0,0758, +0,0888] |
| trần: chọn 2 tốt nhất trong **top-50** | 0,8295 | **+0,1159** CI95 [+0,1083, +0,1235] |

**Dòng top-10 là dòng quan trọng nhất: nó bắt 71% toàn bộ dư địa** (0,0822/0,1159).
Không cần chấm lại sâu cả 50 — sửa đúng phần đầu là ăn gần hết.

Sản xuất đang chọn sai đến mức nào:

```
chọn ĐÚNG cả hai suất trong top-50:   61/1000 =  6,1%
chọn ĐÚNG cả hai suất trong top-10:  171/1000 = 17,1%

hai suất tối ưu nằm ở hạng (theo thứ tự sản xuất):
   hạng 1-2   32,1%      hạng 3-5   12,7%
   hạng 6-10  10,5%      hạng 11+   44,7%
```

44,7% số suất tối ưu nằm từ hạng 11 trở xuống, nhưng vì top-10 đã bắt 71% giá trị,
những suất sâu đó chủ yếu là cải thiện lẻ chứ không phải cú lớn.

> **Ba điều phải đọc kèm, nếu không sẽ dùng sai con số này.**
>
> 1. **Đây là thang harness, không phải thang bảng xếp hạng.** Harness giữ Khối 1 và
>    Khối 3 ở mức gold nên phóng đại hiệu ứng của Khối 2: cùng cấu hình, harness cho
>    0,7136 còn LB private thật là 0,5383. **Không đặt +0,1159 cạnh +0,0337.** Cặp
>    hiệu chuẩn duy nhất đang có (k3/350 → k2/200: harness +0,0487) không phải phép so
>    một biến sạch, nên **không đưa ra hệ số quy đổi** — chỉ biết con số thật nhỏ hơn
>    đáng kể.
> 2. **Trần oracle không ai với tới được.** Nó chọn bằng cách nhìn đáp án chuẩn. Phần
>    lấy được thực tế là một phân số của 0,1159, không phải cả nó.
> 3. **Tham lam là cận dưới.** Suất thứ hai chọn sau khi cố định suất thứ nhất, không
>    vét hết 1.225 cặp. Trần thật ≥ số in ra.

---

## 4. Hai task dạy giống nhau điều gì

Đây là phần có giá trị nhất của việc gộp hai báo cáo lại: bốn kết luận được **hai nhóm
độc lập tìm ra trên hai bài toán khác nhau**.

### 4.1 Giá trị biên của reranker sụp khi rổ tốt lên — tìm ra hai lần

| | Task 1 | Task 2 |
|---|---|---|
| Bằng chứng | rổ trần 0,9117 **>** tầng 1 đầy đủ 0,9100 | nhánh title 64,8% **→** 60,7% sau rerank |
| Đọc | chấm lại 50 ứng viên bằng CE 568M ra kém hơn không chấm | reranker ghi đè thứ tự nền bằng ý kiến đúng ~60% |
| Cách xử | giữ chỗ `blend n` — cho rổ đi thẳng vào đáp án | RRF trọng số α=0,7 — cho reranker **một phần** tiếng nói |

Hai nhóm không trao đổi về việc này. Hai cách chữa khác nhau về hình thức nhưng **cùng
một ý**: đừng để bộ chấm lại có toàn quyền ghi đè một thứ tự nền đã tốt.

### 4.2 Hoà hai danh sách ăn ở khâu **hợp rổ**, không ăn ở khâu **xếp cuối**

| Khâu | Bằng chứng | Kết quả |
|---|---|---|
| hợp rổ ứng viên | Task 1 Part 1: RRF(BM25 ⊕ dense) | **+3,14** R@50 |
| xếp hạng cuối | Task 1 Part 2 #7: RRF/z-score hai thứ hạng | **thua** `blend n=3` ở mọi tham số |
| xếp hạng cuối | Task 2: α-blend dense ⊕ rerank | **+0,006** (kiểm chéo) |

Ba dòng này nhất quán: hoà hai nhánh **bù trừ** nhau (chồng lấp 31–37%) ăn nhiều; hoà
hai thứ hạng của **cùng một tập** ăn ít hoặc âm. Task 2 học điều này theo cách đắt nhất
— bỏ RRF ở khâu hợp rổ và **giết luôn nhánh BM25** mà không ai biết trong sáu tuần.

### 4.3 Kích thước model không mua được gì

| | bằng chứng |
|---|---|
| Task 1 | `mMiniLMv2` 0,118B **hơn** `ViRanker` 0,568B +3,50 điểm, nhanh hơn 16 lần |
| Task 1 | `Qwen3-Reranker-0.6B` (decoder) **bằng** `mMiniLMv2` 0,118B, chậm hơn 40 lần |
| Task 1 | chưng cất từ Qwen3-8B: **thầy 0,5417 thua trò 0,5683** |
| Task 2 | `Qwen2.5-3B` thua `Qwen3-1.7B` ở V4 (0,7648 vs 0,7774) |

Thứ quyết định là **dữ liệu huấn luyện đúng miền**, không phải số tham số. Điều này
cũng khớp với việc đòn lớn nhất của Task 2 là fine-tune (V7→V8 +0,0365), không phải
đổi sang model to hơn.

### 4.4 Chỉ số phụ đơn điệu thì không chọn được tham số

Task 2 mất một vòng thí nghiệm để học: **phủ chữ** tăng đều theo ngân sách nên không
bao giờ có cực trị. Task 1 gặp dạng khác của cùng vấn đề ở công cụ ước lượng
trước-khi-nộp — nó **chỉ dùng được khi hai phía so sánh đến từ cùng một cơ chế chọn**;
áp cho `n=2` vs `n=3` ra +3,49 trong khi thật chỉ +0,14, **hụt 25 lần**, vì một phía
lấy theo rổ còn một phía do CE chọn.

Dạng chung: **đừng đánh giá trên mẫu được chọn bằng một cơ chế khác.**

---

## 5. Bài học phương pháp — gộp từ hai nhật ký

Xếp theo mức tiền đã trả.

1. **Đo trần trước khi chạy.** Task 1 cứu được `M=30` (~4h), chưng cất (~18h) và
   thích nghi `n` chỉ bằng phép đo trần trên CPU. Tổng hệ thống cổng đặt trước cứu
   **~21 giờ GPU**.
2. **Mọi phép đo bộ chấm phải in kèm dòng "rổ trần, không reranker".** Một dòng code,
   cứu 3h GPU và cả một nhánh LoRA.
3. **Ngưỡng phải tương đối so với mốc hiện có**, không bao giờ là con số tuyệt đối bốc
   ra. Task 1 sai ba lần vì đặt ngưỡng tuyệt đối.
4. **Mỗi lượt GPU chỉ đổi một biến, và phải xuất file điểm thô trước khi đóng phiên.**
   Không có file điểm = mọi phân tích hậu kỳ phải chạy lại GPU. Đây là lỗi đắt nhất
   Task 1 từng mắc.
4b. **Một con số không mang danh tính lượt chạy nguy hiểm ngang một file không có md5.**
   Dự án này gắn md5 cho mọi artefact đúng để không so nhầm hai thứ tưởng là một — nhưng
   quy tắc đó phải áp cho cả **con số trích từ nhật ký**. Ca 21/09: ba dữ kiện đều đúng
   riêng lẻ (`4c1cde4` trong HEAD · `models.json` vẫn 256/160 · Khối 3 rỗng 12,1%) được
   ghép thành một cơ chế nghe xuôi, nhưng chúng đến từ **hai lượt chạy khác nhau**.
   Thứ làm chuỗi đó có vẻ tự nhất quán là `retry_used` = 36,8% — **bằng nhau ở cả hai
   lượt**, tức chỉ số **không mang thông tin phân biệt**. Chỉ số phân biệt thật là
   `token_limit_retry_used` (8,6% → 20,0%).
   → Cách bắt: **bảng nhiều nguồn, nhiều cách đếm độc lập**. Không bảng một-nguồn nào
   bắt được. Xem §6.1.
5. **Hai công thức khớp nhau không phải bằng chứng độc lập** nếu chúng ăn cùng dữ liệu
   vào và đều đơn điệu theo cùng một tham số. Bằng chứng phải là khoảng tin cậy hoặc
   tập giữ riêng.
6. **Đổi một tham số làm tham số khác đổi nghĩa.** Task 2: hạ số đoạn 3→2 làm nút cắt
   thật chuyển từ `TARGET_TOTAL` sang ngân sách mỗi đoạn.
7. **Đơn vị đo phải ghi kèm mọi con số.** Cấp Điều ≠ cấp Khoản. Năm mốc METEOR khác
   harness không đặt cạnh nhau được.
8. **Đếm số ca ≠ mức độ mỗi ca.** Chỗ cắt 50.000 ký tự chạm 10,78% văn bản — nghe như
   ít. Nhưng ở ca nặng nhất (văn bản 5,98 triệu ký tự) nó **chỉ giữ lại 0,8% nội dung**.
   Hai trục khác nhau, và comment trong code chỉ nói về trục thứ nhất.
9. **Một comment tự khẳng định con số mà không ai đo thì sống được sáu tuần.** Comment
   "cắt rất ít văn bản" viết 06/08, sai, tồn tại tới 19/09.
10. **Trước mọi lượt chạy trên 1 giờ, hỏi thành viên kia có đang làm việc đó không.**
    Task 1 mất 4h dựng bi-encoder trong khi D đã làm xong. Chi phí hỏi: 2 phút.

---

## 6. Vòng private — đo lại trên code generator mới (20/09/2026, 00:45)

Vòng private Task 2 mở 19/09–23/09, 1.918 câu, `answer: null`.

**Vì sao phải đo lại.** Hiệu số V9 − V8 = +0,0318 được đo trên code generator **cũ**,
khi 12,1% câu có Khối 3 rỗng. Code mới của C (`f55f877`) vá đúng chỗ đó và thêm gom lô
+ hai GPU. Chạy lại toàn bộ trên code sẽ thật sự dùng ở vòng private.

### 6.1 Kết quả — cả hai nhánh, n=1.000, ghép cặp, bootstrap 2.000 lượt seed 42

| | code cũ | code mới `f55f877` |
|---|---:|---:|
| V8 METEOR (bare · α=0 · k=3 · 350) | 0,5144 | **0,5200** |
| V9 METEOR (title · α=0,7 · k=2 · 200) | 0,5462 | **0,5537** |
| **V9 − V8** | +0,0318 [+0,0226, +0,0410] | **+0,0337 [+0,0245, +0,0431]** |
| Khối 3 rỗng — V8 | 101/1000 | **16/1000** |
| Khối 3 rỗng — V9 | 121/1000 | **14/1000** |
| Thời gian V8 | 4h43m | **1h21m** (3,50×) |
| Thời gian V9 | 4h28m | **1h09m** (3,90×) |

**Bản vá có tác dụng thật trên cả hai nhánh** — V8 +0,0056 [+0,0038, +0,0075],
V9 +0,0075 [+0,0052, +0,0100], cả hai CI loại trừ 0.

> ⚠️ **Bẫy đọc số: `retry_used` KHÔNG phân biệt được code cũ với code mới.** Nó bằng
> đúng **36,8%** ở cả hai lượt. Thứ đổi là `token_limit_retry_used`: **8,6% → 20,0%** —
> đó mới là dấu vân tay của `4c1cde4`.
>
> Hệ quả thực tế đã suýt xảy ra 21/09: ghép "12,1% rỗng Khối 3" (số của lượt **cũ**) với
> "retry_used 36,8%" (đúng ở **cả hai** lượt) cho ra kết luận *"cò súng sửa rồi, đạn vẫn
> thiếu"* và một đề xuất nới `max_new_tokens`. Hai tiền đề đều đúng, nhưng lấy từ hai
> lượt khác nhau.
>
> Số thật, ba nguồn hai cách đếm:
>
> | | `conclusion` rỗng | `answer` thiếu K3 | `retry_used` | `token_limit_retry` |
> |---|---:|---:|---:|---:|
> | V9 train code **cũ** | 12,1% | 14,2% | 36,8% | 8,6% |
> | V9 train code **mới** | **1,4%** | 3,5% | 36,8% | 20,0% |
> | **private — bài 0,5383** | **0,9%** | 2,2% | 40,2% | 21,9% |
>
> Giá còn lại của Khối 3 rỗng: **≈ 0,0006**, không phải 0,027.
>
> **Và nới token là hướng đã có bằng chứng âm.** V5 của nhóm generator làm đúng can thiệp
> đó (256→512, retry kết luận 384): METEOR Khối 3 **+0,0787** nhưng toàn bài **−0,0176**.
> T10 đo lại độc lập cùng cơ chế: thêm 59 âm tiết vào đáp án hoàn hảo mất 0,045.
> `configs/models.json` vẫn là 256/160 — **giữ nguyên là đúng, không phải bỏ sót.**

**Nhưng nó không nới rộng khoảng cách V9 − V8 một cách đo được.** Hiệu của hiệu =
+0,0019, CI95 **[−0,0001, +0,0041]** — chứa 0. V9 có nhiều Khối 3 rỗng hơn nên được vá
nhiều hơn một chút, nhưng phần "nhiều hơn" đó không phân biệt được với nhiễu.

> Câu nói được: *bản vá không đổi kết luận V9 > V8, và kết luận đó giờ đứng trên code
> sẽ thật sự chạy.* Câu **không** nói được: *bản vá làm V9 thắng đậm hơn.*

### 6.2 Phép kiểm tương đương của việc gom lô

Chạy thử 40 câu trước khi cam kết hai lượt đầy đủ:

| Phép kiểm | Số |
|---|---|
| Khối 1 khớp **từng byte** giữa code cũ và mới | **40/40** |
| Câu có đáp án lệch | 6/40 — **cả 6 đều là Khối 3 rỗng → có nội dung** |
| `gpu_count` / `batch_size_per_worker` | 2 / 10 — dual-GPU chạy thật |

Gom lô **không** đổi kết quả sinh. Mọi chỗ lệch nằm đúng ở chỗ bản vá nhắm tới. Đây là
bằng chứng mạnh hơn md5 toàn file, vì nó chỉ ra *vị trí* của chỗ lệch. Cơ sở lý thuyết:
cấu hình chạy greedy nên seed không vào RNG, và seed vẫn giữ riêng từng câu trong lô.

### 6.3 Kết quả nộp

| | |
|---|---|
| **Điểm private LB** | **0,5383** |
| Bài nộp | `QA/bai-nop-private/submission.zip`, 1.918/1.918 câu, 0 câu phải điền |
| Thời gian sinh | 7.837 s = **2h11m** — đúng khớp ước lượng trước khi chạy |
| Khối 1 rỗng / Khối 3 rỗng | 0 (0,0%) / 18 (0,9%) |

Ước thời gian đã dựa trên **hai** điểm đo cùng cấu hình (40 câu 237 s · 1.000 câu
4.853 s) nên tách được phần cố định ≈ 45 s, biên 4,08 s/câu. Trên code cũ sẽ là ~8,5 giờ.

**Bộ chấm đã được xác minh trùng khít bộ chấm BTC.** `scoring-btc/scoring.py` gọi
`meteor_score(...)` với tham số mặc định của NLTK (alpha=0,9 · beta=3,0 · gamma=0,5) và
`build_in_tokenizer` là **hàm rỗng** (pyvi đã bị chú thích). Chạy đúng biểu thức của BTC
trên cùng dữ liệu cho **0,553701** — trùng từng số với bộ chấm nội bộ. Nên 0,5537 (train)
và 0,5383 (private LB) **nằm cùng một thang**.

### 6.4 T7 — khoảng cách train → private đến từ đâu

```
METEOR train (offline)   0,5537
METEOR private (LB)      0,5383
khoảng cách             −0,0154
```

Phân bố phía đầu vào của private gần như trùng train:

| | train (n=1.000) | private (n=1.918) |
|---|---:|---:|
| điểm truy hồi top-1 (TB) | 0,5887 | 0,5918 |
| điểm truy hồi top-2 (TB) | 0,4101 | 0,4123 |
| số context/câu | 1,943 | 1,924 |
| độ dài câu hỏi (từ) | 19,92 | 19,62 |
| độ dài context — trung vị | 467 | 462 |
| độ dài context — **phân vị 99%** | 1.585 | **2.381** |

Điểm truy hồi **có** dự báo được METEOR trên train (Pearson r = +0,49; nhóm thấp nhất
0,4129 → nhóm cao nhất 0,6873), nên áp bảng đó lên phân bố private là phép đo có hiệu
lực. Điều kiện hiệu lực theo bài học Task 1 — *chỉ dùng được khi hai phía đến từ cùng
một cơ chế chọn* — được thoả, vì cả hai đi qua cùng pipeline, cùng α, cùng reranker.

| Trục tái trọng số | METEOR private dự đoán | Giải thích được |
|---|---:|---:|
| theo điểm truy hồi top-1 | 0,5540 | **+0,0003** |
| theo độ dài context | 0,5533 | **−0,0004** |

**Private không khó hơn train đối với khâu truy hồi.** Toàn bộ −0,0154 đến từ chỗ khác.

Hai khả năng còn lại, và **không tách được bằng dữ liệu đang có**:

1. **Thiên lệch do chọn tham số** — α, k, ngân sách đều quét trên chính train. Cơ chế
   này đã đo được một lần ở quy mô nhỏ: α quét toàn mẫu +0,0082, kiểm chéo giữ riêng
   +0,006.
2. **Khác biệt phía đáp án chuẩn** — văn phong, độ dài, cách trích luật của private.

Lý do không tách được: **đáp án chuẩn private không nhìn thấy được.** Đúng một tập giữ
riêng thứ hai sẽ tách được, và đó là public — nhưng bảng public **đã khoá** trước khi đo
được. Không thay bằng ước lượng gián tiếp nào.

### 6.5 T8 — chuẩn hoá Unicode: phép đo suýt bị đọc ngược

Phát hiện 4/200 câu đầu không ở dạng NFC. Giả thuyết ban đầu: chuẩn hoá về NFC sẽ khớp
tốt hơn với đáp án chuẩn. Hai cách đo cho **ngược dấu nhau**:

| Kịch bản | Δ METEOR | Dùng được? |
|---|---:|---|
| chuẩn hoá **cả hai bên** | +0,000555 | **không** — ta không chuẩn hoá được đáp án của BTC |
| chuẩn hoá **chỉ bài nộp** (thật) | **−0,000402** CI95 [−0,000814, −0,000076] | có — và nó **hại** |

Lý do: đáp án chuẩn BTC **không đồng nhất** (967/1.000 ở NFC). Khối 2 chép nguyên văn từ
kho luật, và dạng của kho đã khớp sẵn với dạng đáp án chuẩn ở phần lớn trường hợp. Ép về
NFC sửa 11 câu, làm hỏng 15 câu.

> **Bài học:** phép đo phải mô phỏng đúng thứ mình thật sự điều khiển được. Nếu chỉ chạy
> phép đo đối xứng thì đã thấy +0,0006 và đi sửa một artefact đã kiểm sạch, để mất điểm.

### 6.6 Khuyết tật đã biết của khâu đóng gói — không vá, có lý do

14/1.918 câu (0,73%) có **một** context duy nhất dài 16.000–65.000 âm tiết, vượt ngân
sách 200 gấp 80–325 lần. Câu nặng nhất cho đáp án 295.347 ký tự. Lỗi ở khâu **đóng gói**,
không phải generator — chính gói đã chứa context dài như vậy.

Giá thật, đo trên train nơi có đáp án chuẩn:

```
5 câu >20k ký tự   METEOR 0,3061
995 câu còn lại    METEOR 0,5549
tổn thất toàn tập  ≈ 0,0012
```

Không vá, ba lý do: (1) giá 0,0012 nhỏ hơn 20 lần hiệu số V9−V8; (2) khuyết tật đã có
mặt trong lượt train nên 0,5537 đã tính cả nó; (3) file `build_qa_packages_*.py` thuộc
phiên khác theo luật chia file.

**Đáng ghi cho đợt sau:** private có đuôi nặng hơn train rõ rệt (5 câu >50k ký tự, train
có 0). Nên thêm một chặn cứng độ dài context trước khi build.

### 6.7 ⚠️ Tập đánh giá 1.000 câu **đã bị lọc sẵn** — ảnh hưởng cách đọc MỌI số offline

Phát hiện 20/09 khi dựng dữ liệu dạy cho P2.

```
1.000 câu đang dùng để đánh giá : tách được ba khối  1000/1000 = 100,0%
6.000 câu còn lại của train     : tách được ba khối  3203/6000 =  53,4%
```

Danh sách 1.000 `QIDS` được ghi cứng trong `kaggle_layer3/hybrid_rerank_train_notebook.py`.
Chúng được chọn (trực tiếp hay gián tiếp) trong số các câu mà bộ tách khối xử lý được —
harness "cả bài" cần tách được Khối 1 và Khối 3 từ gold mới chạy.

**Hệ quả:** mọi số offline của Task 2 — các bảng ngân sách, T3b, T6, và cả METEOR
train 0,5144 / 0,5537 — đều đo trên **một nửa dân số** có cấu trúc đáp án sạch. Private
không có bộ lọc đó.

Vì sao 46,6% trượt:

| | |
|---|---|
| không có dấu hiệu kết luận | 2.758 (46,0%) |
| không có xuống dòng | 39 (0,7%) |

Trong nhóm thiếu dấu hiệu, **32,8% có đoạn cuối vẫn là trích luật** — đáp án chuẩn dừng
ngay sau Khối 2, không có kết luận. Phần còn lại mở đầu bằng từ khác bộ 7 dấu hiệu
("Trên đây", "Theo quy định", "Trường hợp"…), tức bộ tách chưa đủ chứ không phải đáp án
thiếu khối.

> Đây là quy tắc 3 của Task 1 ở dạng khác: *đừng đánh giá trên mẫu được chọn bằng một cơ
> chế khác*. Cơ chế chọn ở đây là **bộ tách khối của chính ta**.

### 6.8 T10 — giá của việc viết Khối 3 khi đáp án chuẩn không có. **Nhóm đối chứng lật kết luận.**

Giả thuyết nối tiếp 6.7: ta **luôn** sinh Khối 3, mà ~21% câu không cần → mất điểm chính
xác. Đo kiểu "generator hoàn hảo": lấy đáp án chuẩn nguyên vẹn rồi nối thêm **một kết luận
thật do V9 sinh** (trung vị 59 âm tiết).

| Nhóm | METEOR sau khi thêm | Mất so với hoàn hảo |
|---|---:|---|
| **A** — gold **không** có Khối 3 (n=887) | 0,9492 | **−0,0508** CI95 [−0,0545, −0,0471] |
| **B** — gold **có** Khối 3 (n=3.257), đối chứng | 0,9551 | **−0,0449** CI95 [−0,0466, −0,0433] |

**Hai nhóm mất gần như bằng nhau.** Hình phạt đến từ *viết thêm ~59 âm tiết*, không phải từ
*viết kết luận khi không ai cần*. Phần riêng của hiệu ứng cần đo chỉ là **−0,0059**, nhân
21,4% số câu ra **≈ 0,0013** toàn tập. **Đóng hướng.**

> **Nếu chỉ chạy nhóm A thì đã báo 0,0109 và đầu tư theo một con số sai gấp 8 lần.**
> Nhóm đối chứng tốn thêm đúng 4 dòng code. Đây là quy tắc 2 của Task 1 ở dạng khác:
> *mọi phép đo phải in kèm một dòng đối chứng*.

Số 0,0508 vẫn có giá trị riêng, nhưng là một kết luận **khác**: viết dư 59 âm tiết vào một
đáp án vốn đã hoàn hảo làm mất ~0,045–0,05 METEOR ở **mọi** nhóm. Khớp với kết luận V5 của
nhóm generator (bỏ giới hạn 120 từ → Khối 3 tốt lên nhưng toàn bài **giảm**).

### 6.9 T11 — đo trực tiếp nhóm KHÔNG lọc

Nối tiếp 6.7. Nguyên liệu có sẵn, **không tốn lượt rerank nào**: kernel E4 đã nhúng cả
7.000 câu train, nên `outputs/e4/top50_title_qa_train.jsonl` có đủ 7.000 dòng rổ dense
top-50. Dùng α=1,0 để số hạng reranker triệt tiêu khỏi công thức trộn — không cần điểm
reranker. Hai nhóm đi qua đúng một bộ dựng, đúng một cấu hình.

| nhóm | n | METEOR | dài đáp án sinh | K3 rỗng | dài gold (trung vị) |
|---|---:|---:|---:|---:|---:|
| lọc (trong tập đánh giá) | 200 | 0,5429 | 558 | 1,5% | 327 |
| **không lọc** | 200 | **0,5446** | 564 | 0,5% | 268 |

```
không lọc − lọc = +0,0018   CI95 [−0,0382, +0,0423]
```

**Không có hiệu ứng lớn.** Nhưng CI rộng ±0,04 trong khi khoảng cách cần giải thích là
0,0154 — **phép đo không đủ độ phân giải** để xác nhận hay loại trừ một hiệu ứng cỡ đó.

> Câu nói được: *bộ lọc không làm tập đánh giá dễ hơn một cách rõ rệt.*
> Câu **không** nói được: *bộ lọc không ảnh hưởng gì.*

Đáng ghi: nhóm không lọc có gold **ngắn hơn** (268 vs 327) nhưng ta sinh ra dài **bằng
nhau**, nên tỷ lệ dài sinh/gold là 2,27 so với 1,89 — ta viết dư nhiều hơn ở nhóm đó mà
điểm vẫn không thấp hơn. Chưa giải thích được.

---

### 6.10 T12 — reranker đọc tiêu đề Điều (21/09/2026)

**Câu hỏi.** E4 cho tiêu đề Điều vào bộ nhúng dense và thắng. Bộ chấm lại có được đọc tiêu
đề không?

**Tiền đề đã kiểm bằng mã, không tin lời.** Cả ba kernel rerank tạo cặp bằng `u["text"]`
trần:

```
hybrid_rerank_public_notebook.py  :257   pairs.append([q, u["text"]])
hybrid_rerank_train_notebook.py   :265   pairs.append([q, u["text"]])
hybrid_rerank_private_notebook.py :248   pairs.append([qa[qid]["question"], u["text"]])
```

`dieu_tieu_de` không xuất hiện trong file nào. Nhánh "title" của T3 chỉ đổi bộ nhúng dense.

**Thiết kế.** Một rổ (dense-title, cấu hình sản xuất), một lần chạy, hai nhánh khác đúng chữ
đưa vào reranker:

```
rr_bare   →  u["text"]
rr_title  →  dieu_tieu_de + "\n" + u["text"]
```

Tiêu đề qua đúng quy tắc E4: phải tồn tại và khác chính `text` của Điều. Mẫu mới **seed
2109**, 1.000 câu tách được ba khối, **giao với mẫu 1909 = 0** (`outputs/t12_qids.json`).
Kernel: `kaggle_layer3/rerank_title_input_notebook.py`, chạy khoảng 24 phút.

**Chốt kiểm trong kernel — đều qua:**

```
bảng tra tiêu đề         392.691 unit_id · 0 Điều thiếu tiêu đề · 7.163 bị loại (text == tiêu đề)
ứng viên thực sự đổi chữ  48.892 / 50.000 = 97,8%     (sàn đặt trước: 50%)
rr_bare  cặp khác bản trần      0 / 50.000 = 0,0%      ← biến không rò sang đối chứng
rr_title cặp khác bản trần 48.892 / 50.000 = 97,8%     ← biến được áp đúng
rổ 50 ứng viên/câu · 0 text rỗng · md5 hai file khác nhau
reranker xếp khác nhau giữa hai nhánh ở 1.000 / 1.000 câu
```

**Lỗi lúc dựng kernel, bắt được trước khi chạy.** Kernel dựng bằng script sửa kernel cũ.
`\n` trong chuỗi bị nuốt một tầng escape ở năm chỗ, trong đó có chính dấu phân cách tiêu
đề/Khoản. Ba chỗ vẫn là cú pháp hợp lệ nên kernel sẽ chạy được và cho kết quả sai âm thầm.
Bắt bằng `ast.parse` cộng `assert` so dấu phân cách với `chr(92)+chr(110)`. Lúc chấm điểm
heredoc lại nuốt backslash lần nữa — bỏ hẳn `\n` khỏi chuỗi trong lệnh shell. Ghi thành
quy tắc 2.9 ở bản phương pháp.

**Kết quả — METEOR harness, n = 1.000:**

| α | rr_bare | rr_title | chênh |
|---:|---:|---:|---:|
| 0,0 | 0,6822 | 0,7036 | +0,0214 |
| 0,3 | 0,6911 | 0,7076 | +0,0164 |
| 0,5 | 0,6958 | 0,7079 | +0,0121 |
| 0,7 ← sản xuất | 0,6935 | 0,6984 | +0,0049 |
| 0,9 | 0,6888 | 0,6884 | −0,0004 |
| 1,0 | 0,6841 | 0,6841 | **0,0000** |

Chênh lệch giảm đơn điệu theo α và về đúng 0 ở α = 1 — reranker không được dùng thì hai
nhánh buộc phải trùng. Phép tự kiểm nội tại.

**Tách bạch** (ghép cặp, bootstrap 2.000 lượt, **cả hai nhánh cùng α** khi so một biến):

| Thay đổi | Δ | CI95 |
|---|---:|---|
| α 0,7 → 0,5, giữ reranker trần | +0,0023 | [−0,0017; +0,0067] |
| tiêu đề, tại α = 0,5 — một biến | **+0,0121** | [+0,0072; +0,0169] |
| tiêu đề, tại α = 0,7 — một biến | +0,0049 | [+0,0014; +0,0085] |
| tổng: sản xuất → tiêu đề + α = 0,5 | +0,0144 | [+0,0100; +0,0190] |
| kiểm chéo, α chọn trên nửa giữ riêng | +0,0131 | — |

Lần đầu script in "thiên lệch −0,0082" — con số đó so hai thứ khác gốc, nên tách ra đo lại
như bảng trên thay vì tin nó.

**Ngưỡng: 1 trượt, 2 và 3 đạt.**

```
1  METEOR tại cấu hình nộp (α = 0,7) ≥ +0,010      +0,0049   TRƯỢT
2  cận dưới CI95 > 0                               đạt
3  thắng trên nửa giữ riêng                        +0,0131   đạt
```

Bản đăng ký có mâu thuẫn nội tại: ngưỡng 1 khoá α ở giá trị chỉnh cho reranker **cũ**,
trong khi cùng bản đó đã đoán α tối ưu sẽ dịch khi reranker giỏi lên. Lỗi thiết kế nằm ở
người đặt ngưỡng, phát hiện sau khi có số. Ghi thành quy tắc 2.7 ở bản phương pháp.

**Quyết định: triển khai, ghi rõ là phán đoán sau khi thấy số, không phải "đạt ngưỡng".**
Lý do: CI loại trừ 0 ở cả hai mức α, có cơ chế, có tự kiểm ở α = 1, α = 0,5 chọn bằng kiểm
chéo; và theo Điều 5 bản tốt nhất được tính nên lượt mới không thể làm mất 0,5383.

**Trạng thái lúc ghi (21/09, khoảng 04:35 UTC).** Kernel rổ private đã dựng
(`kaggle_layer3/rerank_private_title_input_notebook.py`, 04:28). Lượt private đổi đúng hai
thứ so với bài 0,5383: chữ đưa vào reranker, và α 0,7 → 0,5. Chi phí ước: rổ private ~45
phút, build ~2 phút, sinh 1.918 câu ~2 giờ 11 phút. **Điểm LB: chưa có.**

Dự đoán đặt trước: LB mới − 0,5383 trong khoảng **+0,003 đến +0,012**. Quanh 0 thì harness
không chuyển được, không mất gì. Âm rõ thì kiểm build: α có đúng 0,5, reranker private có
thật đọc tiêu đề không.

**Chưa đo:** số cặp vượt giới hạn 512 token của reranker ở mỗi nhánh.

---

## 7. Kế hoạch chạy tuần tự

Xếp theo **giá trị chẩn đoán trước, giá trị điểm sau**, và theo quy tắc 1 của Task 1:
làm hết phần CPU trước khi chạm GPU.

### Bước 0 — cửa quyết định, làm trước mọi thứ (10 phút, không code)

**Vòng private có nhận nộp lại không?** Câu trả lời đảo thứ tự cả kế hoạch:

- **Có** → còn ~3 ngày (tới 23/09). Chỉ P1 và P4 kịp. P2 hoãn sang sau.
- **Không** → mọi thứ dưới đây là cho báo cáo và vòng sau; chạy theo đúng thứ tự P1→P4.

Đồng thời: **thu hồi và tạo lại token Kaggle** đang dùng (nó đã từng bị dán vào khung chat).
Và dừng kernel `legalqa-public-v9-title-k2` nếu còn chạy — CLI không có lệnh cancel, phải
dừng trên trang web.

---

### ~~P1 — Chấm lại chỉ top-W~~ · ✅ ĐÃ CHẠY 20/09 · **TRƯỢT CẢ BA NGƯỠNG → ĐÓNG**

Giả thuyết: T6 đo 71% dư địa nằm trong top-10, T3 đo reranker **làm hại** khi ghi đè cả 50
→ nên giữ nguyên thứ tự nền ngoài top-W, chỉ trộn α trong top-W. Quét W × α trên
`L3_rerank_train_title.jsonl`. Script: `pipeline/eval_cua_so_rerank.py`.

| W \ α | 0,5 | 0,6 | 0,7 | 0,8 | 0,9 | 1,0 |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0,7152 | 0,7154 | **0,7156** | 0,7130 | 0,7099 | 0,7040 |
| 10 | 0,7134 | 0,7138 | 0,7137 | 0,7135 | 0,7099 | 0,7040 |
| 20 | 0,7126 | 0,7136 | 0,7136 | 0,7135 | 0,7099 | 0,7040 |
| **50** ← hiện tại | 0,7127 | 0,7136 | **0,7136** | 0,7135 | 0,7099 | 0,7040 |

| Ngưỡng đặt trước | Kết quả | |
|---|---:|---|
| lợi ≥ +0,010 | +0,0019 | **KHÔNG** |
| cận dưới CI95 > 0 | −0,0003 | **KHÔNG** |
| thắng trên nửa giữ riêng | −0,0002 | **KHÔNG** |

```
bootstrap toàn mẫu   +0,0019
kiểm chéo hai nửa    −0,0002      (nửa 1 chọn W=5 α=0,7 -> +0,0014
thiên lệch do chọn   +0,0021       nửa 2 chọn W=5 α=0,5 -> −0,0018)
```

**Thiên lệch do chọn lớn hơn chính hiệu ứng.** Đúng như lo ngại khi đặt ngưỡng: quét 24 ô
thì bootstrap toàn mẫu không đọc được.

**Vì sao nó trượt — đây mới là phần đáng giá.** Cột α=1,0 (dense thuần) = 0,7040 ở **mọi**
W. Nên đóng góp thật của reranker là +0,0096, và nó đã bị α=0,7 ghìm sẵn: với trọng số
nghịch đảo hạng, một ứng viên ở dense_rank 40 **không thể** bị reranker kéo lên đỉnh. Cắt
cửa sổ W chỉ làm lại đúng việc mà α đã làm.

> Đây là quy luật 7 của Task 1 lặp lại nguyên văn: *"bốn giải pháp hoá ra là một, đừng
> cộng dồn chúng."* α và cửa sổ W **sửa cùng một chỗ bằng cùng một cơ chế**.

**Hệ quả cho T6 — phải đọc lại trần 0,1159 cho đúng.** Trần đó nói *thông tin để chọn đúng
có nằm trong rổ*. Nó **không** nói *hai tín hiệu hiện có chứa thông tin đó*. P1 vừa chứng
minh vế sau là sai: xáo lại bằng chính dense và rerank, ở bất kỳ cửa sổ nào, không lấy
được gì. Muốn chạm vào 0,1159 phải có **tín hiệu mới** — tức P2, không phải xáo lại.

---

### ~~P2 — Fine-tune bi-encoder mức Khoản~~ · ✅ ĐÃ CHẠY 20–21/09 · **TRƯỢT NGƯỠNG → ĐÓNG**

Dữ liệu dạy: 1.756 triplet (câu hỏi ↔ Khoản ↔ Khoản khác **cùng Điều**), giữ riêng 439.
`CachedMultipleNegativesRankingLoss`, lô 32 mini-batch 4, LR 1e-5, 3 epoch, fp16.
Đo trên 439 câu giữ riêng, embedding gốc lấy từ kernel E4.

| | gốc | fine-tune | Δ (điểm) | CI95 |
|---|---:|---:|---:|---|
| recall@1 | 0,2301 | 0,2346 | +0,46 | [−2,96, +3,87] |
| recall@5 | 0,4806 | 0,5103 | +2,96 | [−0,68, +6,61] |
| recall@10 | 0,5809 | 0,5854 | +0,46 | [−2,73, +3,87] |
| **recall@50** | **0,7403** | **0,7380** | **−0,23** | [−2,96, +2,51] |

Ngưỡng đặt trước (bằng ngưỡng E4): recall@50 ≥ +2,0 **và** cận dưới CI > 0 → **trượt cả hai**.

recall@5 +2,96 là chỉ số đẹp nhất trong bốn, cận dưới CI vẫn âm. Chọn nó là đúng thiên
lệch mà T9 vừa dạy. Không dùng.

> **Khuyết điểm thiết kế phải ghi, kèm con số để lần sau dùng được.**
>
> n=439 cho CI nửa-rộng ±2,7 điểm, mà ngưỡng đặt là +2,0. Nếu hiệu ứng thật đúng bằng
> +2,0 thì CI sẽ là [−0,7, +4,7] — **chứa 0**, tức không xác nhận được chính cái ngưỡng
> mình đặt ra.
>
> Cỡ mẫu cần, suy từ chính CI đã đo (nửa-rộng tỷ lệ nghịch với √n):
>
> | mục tiêu | n cần | CI nửa-rộng |
> |---|---:|---:|
> | vừa đủ để CI loại trừ 0 khi hiệu ứng = +2,0 | **≈ 830** | 2,0 |
> | đủ thoải mái, CI [+0,5, +3,5] | **≈ 1.500** | 1,5 |
>
> *(Ước bằng tỷ lệ không ghép cặp `√(0,74×0,26/n)` cho ~1.500–2.000; con số đó **lớn hơn
> thực tế** vì phép so này ghép cặp — chỉ câu bất đồng mang thông tin. Dùng CI đã đo để
> ngoại suy thì đúng hơn.)*
>
> Có sẵn 6.000 câu train chưa lọc, nên n=1.500 là làm được. **Lần sau: chọn n TỪ ngưỡng,
> đừng chọn ngưỡng rồi mới lấy n có sẵn.**
>
> Quan sát −0,23 cách ngưỡng đủ xa nên kết luận vẫn vững. Chỗ chết là nếu nó rơi vào
> +1…+3.

**Ba lỗi dữ liệu đã bắt trước khi tiêu GPU** (đáng giá hơn kết quả):

| Lỗi | Phát hiện bằng | Sửa |
|---|---|---|
| Chọn nhãn bằng **recall** → chọn Khoản **dài nhất** | độ dài trung vị 334 âm tiết; một Khoản gán cho **38** câu | đổi sang **F1** → trung vị 119, tối đa 5 |
| Không có negative **cùng Điều** | công thức Task 1 chỉ có negative trong lô | thêm → 35,6% negative cùng Điều |
| Positive/negative lệch độ dài 6 lần | bộ phân loại **chỉ dùng độ dài** đạt **94,3%** | cân bằng độ dài → 68,6%, dưới mức **tự nhiên 74,2%** |

> Lỗi thứ nhất **chính là bẫy E6** đã ghi trong báo cáo này — *phủ chữ đơn điệu tăng theo
> độ dài nên không có cực trị*. Bài học được ghi cho việc **chọn tham số**; nó cũng đúng
> cho việc **chọn nhãn**, và hai chuyện đó đã không được nối lại.
>
> Mức "tự nhiên 74,2%" đo trên 18.177 cặp (gold vs mọi Khoản khác cùng Điều). Cổng 65%
> tôi đặt ban đầu là **con số tuyệt đối bốc ra** — đúng thứ quy tắc 1 của Task 1 cấm.

### ~~P2 (bản kế hoạch gốc)~~ — giữ để đối chiếu

Task 1 ăn **+1,90 điểm bảng thật** từ đúng việc này — đòn lớn nhất cả tháng của họ. Cùng
model `AITeamVN/Vietnamese_Embedding` nên công thức chuyển thẳng:

```
CachedMultipleNegativesRankingLoss · lô 64, mini-batch 8 · LR 1e-5 · warmup 10% · fp16
```

Chỉ đổi dữ liệu dạy: cặp (câu hỏi ↔ **Khoản**) thay vì (câu hỏi ↔ văn bản).

**Chặn:** cần cặp (câu hỏi, khoan_id) sạch. `data123/` vẫn chưa có. Nguồn thay thế là nhãn
theo chuỗi (Khoản phủ nhiều chữ nhất Khối 2), chấp nhận nhiễu — chính là cách E3a/E5/E6b
đã dùng.

> **Đừng mượn `vi_emb_ft/` rồi kỳ vọng.** Nó được dạy kéo cả văn bản lại gần câu hỏi;
> ta cần **tách các Khoản trong cùng một văn bản ra khỏi nhau**. Ở mức chi tiết, hai mục
> tiêu gần như đối nghịch. Thử được vì miễn phí, nhưng **đặt kỳ vọng ở mức có thể âm**.

**Ngưỡng đặt trước:** recall@50 ≥ +2,0 điểm **và** cận dưới CI > 0 — cùng ngưỡng E4.

---

### P3 — Khôi phục RRF ở khâu hợp rổ · CPU + 1 lượt GPU nhẹ

Task 1 Part 1 ăn **+3,14 R@50** từ đúng việc này (chồng lấp BM25↔dense chỉ 31–37%). Task 2
hiện hợp rổ rồi xếp thuần bằng dense, nên **về cấu trúc BM25 không đóng góp được gì** —
trùng lặp rổ-cuối ↔ dense top-50 là 96,2%.

**Nhưng xếp sau P1/P2**, vì nó sửa *sự có mặt* mà T1 đã đo sự có mặt **không** phải nút
thắt: gold Điều ngoài top-50 chỉ 9,6–13,7%.

Nếu làm thì đi kèm: vá `MAX_TOKENIZE_CHARS` (BM25 đang mù 25,39% kho). Hai việc này **chỉ
có nghĩa cùng nhau** — vá lỗ cắt mà không khôi phục RRF thì không chảy vào điểm.

---

### P4 — Chặn cứng độ dài context trước khi build · 30 phút

Vá khuyết tật 6.6. Giá thấp (0,0012) nhưng rẻ và private có đuôi nặng hơn train rõ rệt.
Thuộc `build_qa_packages_*.py` — **phiên khác sở hữu file này**, phải báo trước.

---

### ✅ T12 — reranker đọc tiêu đề · ĐÃ CHẠY 21/09 · ngưỡng 1 trượt, 2 và 3 đạt → TRIỂN KHAI (phán đoán)

Xem 6.10. Lượt private đang chuẩn bị; ghi điểm LB vào 6.10 khi có.

---

### D — Qwen3-1.7B làm giám khảo cho top-10 · chưa chạy

Hỏi model "Khoản này có trả lời câu hỏi không", lấy điểm ở token đầu, trộn như tín hiệu thứ
ba trong top-10. Nhắm đúng nút thắt T6 (71% dư địa trong top-10) bằng một tín hiệu **mới** —
thứ T9 chứng minh đang thiếu. Kỳ vọng thấp vì model nhỏ. Kế hoạch đầy đủ:
[`KE_HOACH_THI_NGHIEM_D_E.md`](KE_HOACH_THI_NGHIEM_D_E.md) mục 2.

---

### E — HyDE · chưa chạy

Nhúng một đoạn luật giả do Qwen viết thay vì (hoặc trộn với) câu hỏi. Nhắm vào **sự có mặt**,
mà sự có mặt không phải nút thắt (T1), nên có một cửa rẻ trước: recall@50 cấp Điều phải tăng
≥ +2,0 điểm mới được chạy cả đường ống. Kế hoạch đầy đủ:
[`KE_HOACH_THI_NGHIEM_D_E.md`](KE_HOACH_THI_NGHIEM_D_E.md) mục 3.

Cả D và E dùng mẫu mới **seed 2209**, không giao với 1909 (đã chỉnh α, k, ngân sách) và 2109
(đã chọn α = 0,5).

---

### Đã đóng — không đưa vào kế hoạch

| Hướng | Vì sao |
|---|---|
| Chấm cả hai vector bare+title lấy max | sửa 75 câu thua của E4, nhỏ so với trần 0,1159 |
| Quét lại ngân sách / số đoạn | trục đã đóng, CI loại trừ 0 |
| Nở đoạn ra cả Điều | METEOR tụt dù phủ chữ tăng 19 điểm |
| Chuẩn hoá NFC bài nộp | đo được **hại** −0,0004, CI loại trừ 0 |
| Ưu thế toàn cục theo loại văn bản | Task 1 đã thử, **âm**, kể cả khi ước tham số ngay trên dev |

---

## 8. Nguồn

| Nội dung | File |
|---|---|
| Task 1 — nhật ký kỹ thuật, hướng đã đóng, quy luật | `IR/part2-xep-hang/docs/NHAT_KY_KY_THUAT.md` |
| Task 1 — bảng 10 bộ chấm | `IR/part2-xep-hang/docs/bang_ablation.md` |
| Task 1 — sinh ứng viên | `IR/docs/Trang_thai_Part1_05-09-2026.md` · `IR/part1-sinh-ung-vien/README.md` |
| Task 2 — truy hồi, E1–E6b/T1–T5 | `QA/part1-truy-xuat/retrieval/docs/BAO_CAO_THI_NGHIEM_RETRIEVAL.md` |
| Task 2 — mục lục quyết định một dòng/thí nghiệm | `QA/part1-truy-xuat/retrieval/docs/NHAT_KY_THI_NGHIEM_RETRIEVAL.md` |
| Task 2 — bằng chứng chi tiết từng thí nghiệm | `QA/part1-truy-xuat/retrieval/docs/KET_QUA_THUC_NGHIEM_18-09.md` |
| Task 2 — generator V2–V7 | `QA/part2-sinh-cau-tra-loi/reports/baseline_v{2..7}_*.md` |
| Task 2 — phân rã mất điểm, bảng version | `QA/CLAUDE_1.md` |
| Bản đồ hai nhánh | `README.md` · `QA/README.md` |

Bốn repo gốc: `justySusanto/Task1_Retrieval` · `Loc2116/DSC_2026_Task1_Part2` ·
`maidang09122006-ux/ChanTaooDe-` · `Hakuchoiii2/LegalQA_Task-2`.

**Dữ liệu BTC cấp không được tái phát tán.** Không repo nào chứa dữ liệu; mọi dataset
và kernel Kaggle của dự án đều đặt Private.
