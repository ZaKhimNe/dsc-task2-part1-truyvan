# Kết quả thực nghiệm E1–E6b — 18/09/2026

Ghi theo khuôn §7 của `PLAN_TOI_THUONG_RETRIEVAL_v2.md`: ngưỡng đặt trước → kết quả → n →
quyết định → file bằng chứng.

## Điều kiện đo — ĐỌC TRƯỚC KHI DÙNG BẤT KỲ SỐ NÀO

Máy chạy là **bản clone mới** (`QA/part1-truy-xuat/`), tại thời điểm đo **không có**:
`outputs/layer2/`, `outputs/layer3/`, `b0_labels_*`, `citation_labels_*`, và **không có
`data123/`**. Nên mọi phép đo dưới đây dùng:

- **Nhãn theo chuỗi**: gold = Khối 2 tách khỏi `reference_answer` bằng marker kết luận
  (`split_blocks`), rồi so bằng phủ chữ / METEOR trên TEXT. Không dùng `unit_id`.
- **Rổ thế chân**: BM25 top-20 văn bản (có `bm25_doc_index.pkl`), không có dense/rerank.

> **KHÔNG trộn các số ở đây với `gold gộp n=286`** (B0 + citation) của plan §1.4. Hai định
> nghĩa gold khác nhau, cho Hit@3 khác nhau (39,5% vs ~60%) — plan §8 bẫy 13 đã cảnh báo.

Nhãn theo chuỗi có một ưu thế riêng: **miễn nhiễm bug mã trùng**. Đo theo mã thì rổ trả đúng
`unit_id` vẫn tính là trúng dù text giao ra là khoản con anh em; đo theo chữ thì không.

Tỷ lệ tách được 3 khối ≈ 56% số câu — phần không tách được bị loại khỏi mọi mẫu.

---

## Bảng tổng hợp

| Thí nghiệm | Ngưỡng đặt trước | Kết quả | n | Quyết định | File bằng chứng |
|---|---|---|---|---|---|
| **E1** thiệt hại bug mã trùng | — (đo để quyết E2) | 100% bản sao **cùng một Điều**; 95,6% khác text, Jaccard median 0,107 | 432.473 hàng | **E2 xuống cuối** | `pipeline/measure_dup_unit_id_damage.py` |
| **E3a-bm25** trần chunking | — (đo để quyết mở rộng) | Khoản trần phủ 0,739 · cả Điều 0,915; một Khoản đủ chỉ **8,7%** số câu | 300 | Mở rộng sang Điều là đúng hướng | `pipeline/eval_chunking_ceiling_string.py` |
| **E3a-heldout** trần chunking | phủ chữ tăng ≥ 10 điểm | 63,5% → 82,5% (+19,0); 354 tốt hơn / 1 xấu hơn | 472 | Chạm ngưỡng, nhưng không nở cả Điều (xem E6b) | **không có script trên máy này** |
| **E6** quét ngân sách (phủ chữ) | — | đỉnh 300, nhưng **hỏng** do thiên vị độ dài khi chọn Khoản xuất phát | 300 | Bỏ, thay bằng E6b | (gộp vào E6b) |
| **E6b** quét ngân sách (METEOR, 3 bộ chọn) | — | đỉnh **400–500**, vùng phẳng 300–500; cả Điều **thua** ở cả 3 bộ | 300 × 3 | **Giữ `PER_ITEM_BUDGET_SYLLABLES=350`, đóng trục** | `pipeline/eval_budget_sweep_meteor.py` |
| **E5** đọc dẫn chiếu | ≥ +2,0 recall@50, CI dương | +1,7 điểm% (ngưỡng phủ 90%); METEOR +0,0121; 8 thắng/1 thua | 300 | **KHÔNG ĐẠT — treo**, chờ biến thể top-5 | `pipeline/eval_citation_following.py` |
| **E4** nhúng kèm tiêu đề | ≥ +2,0 điểm & CI dưới > 0 | phủ ≥80%: **+4,40đ**; ≥90%: **+4,50đ**; ≥95%: **+2,50đ**; METEOR **+0,0217** | 1.000 | **ĐẠT — dùng nhánh title** | `pipeline/eval_e4_title_embedding.py` |

---

## E1 — Bug mã trùng `retrieve.py:413`

```
Tổng hàng (= hàng ma trận embedding) : 432.473
Mã duy nhất                          : 393.998
Hàng bị che                          :  38.475 = 8,9%
Mã bị trùng                          :   7.631
  ├─ cùng một Điều                   :   7.631 = 100,0%
  ├─ khác Điều, cùng văn bản         :       0
  └─ KHÁC VĂN BẢN                    :       0
Text hàng đầu vs hàng cuối: khác nhau 95,6%, Jaccard median 0,107, rời hẳn 9,9%
```

**Kết luận sửa lại chẩn đoán ban đầu.** Các dòng bị che **không vô hình** với dense search:
`dense_search_units` chấm điểm toàn bộ N hàng rồi map `vị trí → uid`, chiều đó không hỏng.
Chỗ hỏng là ba nơi giải mã cùng một mã ra ba hàng khác nhau:

| Bước | Lấy hàng nào |
|---|---|
| `dense_search_units` tìm ra | hàng **i** (khớp thật) |
| `rerank_units_dense` chấm lại | hàng **cuối** mang mã đó (qua bảng tra dòng 413) |
| `get_unit_text` lấy text giao C | hàng **đầu** mang mã đó |

Vì bản sao **luôn cùng một Điều**, và V8 đã gộp Khoản liền kề cùng Điều, thiệt hại phần lớn
bị hấp thụ. Suy luận "9,4% ≈ một phần ba của 27,3% đang thiếu" **không đứng được**.

**Cách sửa nên dùng:** đánh số mã cho duy nhất lúc build (`uid`, `uid#2`, …) theo đúng thứ tự
duyệt của notebook nhúng. KHÔNG dùng `{uid: [danh sách vị trí]}` + lấy max — lấy max trên k
bản sao là thống kê thứ tự, mã `280203_23_1` (852 bản) sẽ gần như luôn thắng suất top-50
trước unit trung thực chỉ có 1 hàng.

Con số `403.604` từng dùng là đếm **chỉ Khoản**; `432.473` là toàn bộ danh sách unit mà
`build_corpus_embedding_index` nhận — đó mới là mẫu số đúng cho bug này. Đã được output
CELL 5 của kernel Kaggle xác nhận độc lập.

## E2 — Vá bug mã trùng (18/09, sau khi có rổ thật của E4)

### Chỗ thứ tư, phát hiện muộn

Ba chỗ đầu đã ghi ở E1. Chỗ thứ tư tìm ra khi đọc lại code để vá, và nó **nặng hơn cả ba**:

`union_units` ([retrieve.py](../src/b2_retrieval/retrieve.py)) loại trùng theo `unit_id`, giữ
bản **gặp đầu tiên**. Nên hai khoản con khác nhau cùng mang một mã chỉ chiếm được **một suất**
trong rổ — bản kia bị vứt dù dense đã chấm điểm riêng cho nó. Và vì `units_lexical` được
truyền vào trước, bản BM25 luôn thắng suất, bản dense luôn thua.

> **Đính chính.** Trước đó tôi viết "các dòng bị che KHÔNG vô hình với dense search". Đúng ở
> mức `dense_search_units`, **sai ở mức rổ**: qua `union_units` thì chúng bị gộp mất.

Bốn chỗ giải cùng một mã ra bốn hàng khác nhau:

| Bước | Lấy hàng nào |
|---|---|
| `dense_search_units` | hàng **i** — hàng khớp thật |
| `union_units` | hàng **gặp đầu tiên**; các hàng còn lại **bị vứt** |
| `rerank_units_dense` | hàng **cuối** mang mã đó |
| `get_unit_text` | hàng **đầu** trong `parsed_corpus` |

### Cắn bao nhiêu trong rổ thật (`measure_dup_impact_in_basket.py`, 7.000 câu, top-50)

| Nhánh | Suất mất/50 (mean) | median | p90 | ≥1 suất | ≥5 suất | tệ nhất |
|---|---:|---:|---:|---:|---:|---:|
| bare | 0,55 | 0 | 2 | 22,5% | 3,0% | 38/50 |
| **title** | **0,99** | 0 | 3 | 26,3% | 6,6% | 40/50 |

**Nhánh title mất gần gấp đôi nhánh bare.** Ghép tiêu đề làm các khoản con cùng Điều giống
nhau hơn → xúm vào top-50 cùng lúc → va nhau nhiều hơn. Nghĩa là **E4 thắng +4,40/+4,50 trong
khi đang chịu thiệt hại lớn hơn đối thủ**, tức con số thắng của nó đang bị ghi thiếu, và vá
bug sẽ cộng thêm cho đúng nhánh sắp dùng.

Trung vị bằng 0 — phần lớn câu không hề hấn. Nhưng đuôi nặng: 6,6% số câu mất ≥5 suất, câu
tệ nhất chạy với 10/50 ứng viên.

### Bản vá — cộng thêm, KHÔNG đổi hành vi mặc định

Phiên kia đang chạy song song nên mọi thay đổi đều giữ mặc định nguyên trạng:

| Hàm | Sửa gì |
|---|---|
| `build_corpus_embedding_index` | thêm `warn=True` — **in ra** số hàng bị che thay vì đè im lặng. Vẫn giữ hàng cuối, hành vi không đổi |
| `dense_search_units` | gắn thêm `row` (vị trí hàng thật) vào mỗi `UnitCandidate` |
| `rerank_units_dense` | **ưu tiên `row`** nếu có; chỉ tra bảng hỏng khi không có (nhánh BM25) |
| `union_units` | thêm `dedup_by_row=False` — mặc định y như cũ; bật lên thì khoá là `row`, giữ được các khoản con anh em |

Kiểm chứng (chạy được lại):

```
index giữ hàng cuối           -> 4      (hành vi cũ, không đổi)
union mặc định: gộp còn       -> 1 suất (hành vi cũ, không đổi)
union dedup_by_row=True       -> 3 suất (vá)
rerank có row  -> chấm hàng 1 -> 1.0    (đúng đoạn đã khớp)
rerank không row -> bảng tra  -> 0.0    (lùi về hành vi cũ)
```

### Bật `dedup_by_row=True` thì được gì? (`eval_dedup_by_row_gain.py`, n=1.000/nhánh)

| Nhánh | Suất mất/50 | Phủ chữ | METEOR | CI 95% | Thắng/thua |
|---|---:|---:|---:|---|---:|
| bare | 0,58 | +0,0001 | +0,0003 | [−0,0001, +0,0009] | 4 / 3 |
| title | 1,04 | +0,0006 | +0,0014 | [+0,0000, +0,0032] | 5 / 4 |

**Gần như không gì.** BARE có CI chứa 0. TITLE cận dưới đúng +0,0000 — sát mép, và chỉ 9/1.000
câu đổi kết quả.

**Vì sao bằng 0 — ba phát hiện tự khoá vào nhau:**

- E1: bản sao **luôn cùng một Điều** (100%, 0 xuyên văn bản)
- E6b: ngân sách mở rộng **400 âm tiết** → gộp các Khoản liền kề cùng Điều
- ⇒ khoản con bị `union_units` vứt đi **vốn đã nằm trong đoạn giao ra**, qua bước gộp

Rổ mất suất, nội dung không mất.

### QUYẾT ĐỊNH: giữ bản vá, KHÔNG bật cờ. E2 ĐÓNG.

Giữ bản vá vì nó là vệ sinh có giá trị riêng: hết đè im lặng, và `rerank_units_dense` chấm
đúng hàng đã khớp khi có `row`. Không bật `dedup_by_row=True` vì lợi đo được nằm trong nhiễu,
mà bật lên là đổi kích thước rổ — thêm một biến không đáng.

> **ĐÓNG CÓ ĐIỀU KIỆN.** Kết luận "vô hại" này phụ thuộc `PER_ITEM_BUDGET_SYLLABLES = 400`.
> Hạ ngân sách về 0 (giao Khoản trần) thì không còn bước gộp để vớt lại, union collapse sẽ
> cắn thật. **Đổi ngân sách → đo lại E2.**

## E3a — Trần chunking (HAI phép đo riêng, KHÔNG so được với nhau)

| Mã | Rổ | n | Phủ chữ: Khoản → cả Điều | Nguồn |
|---|---|---:|---|---|
| E3a-heldout | rổ thật, k=3 | **472** | 63,5% → 82,5% (+19,0) | chỉ còn trong bản ghi |
| E3a-bm25 | BM25 top-20 | 300 | 73,9% → 91,5% (+17,6) | `eval_chunking_ceiling_string.py` |

> **Hai mốc khởi đầu khác nhau nên hai mức tăng KHÔNG so được.** 63,5% so với 73,9% chênh
> nhau vì 20 đoạn đương nhiên che nhiều chữ hơn 3 đoạn. Nên +19,0 và +17,6 **không phải**
> "hai lần đo cùng một hiệu ứng ra gần nhau" — chúng là hai hiệu ứng trên hai bài toán khác
> độ khó. Đừng lấy trung bình, đừng coi cái này xác nhận cái kia.
>
> Dòng `n` của E3a-heldout từng bị ghi nhầm là "1.125 context_id" — đó là **số văn bản**,
> không phải số câu. Số câu thật là **472**. Đã sửa 18/09.

### E3a-bm25 — chi tiết

| Phủ chữ Khối 2 | Khoản đơn tốt nhất | Cả Điều tốt nhất |
|---|---:|---:|
| mean | 0,739 | 0,915 |
| ≥80% | 40,0% | 88,0% |
| ≥90% | 18,3% | 67,3% |
| ≥95% | 8,3% | 49,0% |

Điều hơn Khoản ≥10 điểm% (chunking Khoản làm vỡ đáp án): **72,7%**. Một Khoản đã đủ: **8,7%**.

### E3a-heldout — chi tiết

Giá phải trả khi nở cả Điều: độ dài context median **333 → 1.098 âm tiết** (gấp 3,3 lần),
trong khi `LEN_CAP = 1200` — median đã gần chạm trần, p90 chắc chắn vượt. Đây là lý do
độc lập thứ hai (cạnh METEOR ở E6b) để không nở cả Điều thẳng tay.

**Sai sót đã sửa trong lúc chạy:** lần đo đầu dùng kiểm tra chuỗi-con, ra −22,7 điểm. Chuỗi-con
chỉ đúng khi đoạn ứng viên NGẮN hơn gold; nở cả Điều thì nó dài hơn. Đo lại bằng phủ token.

Repo ghi trần chunking **31,1%** (`measure_chunking_ceiling.py`) — chênh gấp đôi vì script đó
chạy trên nhãn B0-confident, đúng bộ thiên về câu trích gần nguyên văn một Khoản.

**Cảnh báo phương pháp:** phủ chữ tính bằng tập hợp nên text dài **không thể thua** — Điều ⊇
Khoản. Vì vậy +48 điểm% là **cận trên**, không phải mức lợi thật.

## E6b — Quét `PER_ITEM_BUDGET_SYLLABLES`, chấm bằng METEOR

| Bộ chọn Khoản xuất phát | Khoản xuất phát (median) | Đỉnh | METEOR đỉnh | Khoản trần | Chênh |
|---|---:|---:|---:|---:|---:|
| A recall (thiên vị Khoản **dài**) | 468 | 300 | 0,4658 | 0,4194 | +0,046 |
| **B f1 (trung hoà)** | **145** | **500** | **0,5478** | 0,4269 | +0,121 |
| C precision (thiên vị Khoản **ngắn**) | 26 | 400 | 0,4538 | 0,2358 | +0,218 |

*(median Khoản thật của corpus = 49 âm tiết)*

**Phát hiện phương pháp quan trọng:** phủ chữ **đơn điệu tăng** theo ngân sách ở cả 3 bộ
(B: 0,665→0,784), trong khi METEOR **có đỉnh** rồi tụt. Chọn ngân sách theo phủ chữ sẽ luôn
trả về "ngân sách lớn nhất còn lọt `LEN_CAP`" — không có cực trị để tìm. **Mọi quyết định về
lượng text phải chấm bằng METEOR**, vì nó phạt chữ thừa và phạt đứt đoạn.

"Cả Điều" thua đỉnh ở **cả ba** bộ chọn (A 0,4290 / B 0,5247 / C 0,4466). Không trả cả Điều.

Đỉnh phẳng 300–500 (bộ B chênh 0,0036 trên cả dải). **`PER_ITEM_BUDGET_SYLLABLES = 350` của
V8 nằm trong vùng phẳng ở cả ba bộ** → giữ nguyên, đóng trục ngân sách. Đổi 350→400 là hợp lệ
nhưng lợi dưới nhiễu.

Cột "dài median" bão hoà từ ngân sách 400 trở đi (hết Khoản anh em để gộp) — thêm một lý do
không quét tiếp.

## E5 — Đọc dẫn chiếu

```
Tham chiếu cùng văn bản : 133.186 (91,9%)
Tham chiếu khác văn bản :  11.721 (8,1%) — giải được số hiệu: 6.564 = 56,0%
Điều thêm vào rổ/câu    : median 17

Với tới được  phủ ≥70%: 93,0% → 93,7% (+0,7 điểm%)
              phủ ≥90%: 52,3% → 54,0% (+1,7 điểm%)
METEOR (ngân sách 400) : 0,4242 → 0,4363  (+0,0121)
                         8 tốt hơn · 1 tệ hơn · 291 không đổi
```

**Không đạt ngưỡng +2,0 của plan P5.** 91,9% dẫn chiếu trỏ trong cùng văn bản nên cơ chế chủ
yếu vớt lại Điều rơi khỏi top-50, không mở rộng sang văn bản BM25 bỏ sót. 44% tham chiếu khác
văn bản không giải được số hiệu (một phần trỏ tới luật ngoài corpus đã lọc) — trần cứng.

Phép đo này còn **ưu ái E5**: chọn theo max phủ chữ trên toàn rổ đã nở, tức cho nó lợi thế
oracle; sản xuất phải để 17 Điều thêm vào chen trong top-50 của reranker. Nên +0,0121 là cận
trên.

**Biến thể đáng thử trước khi đóng hẳn:** chỉ đọc dẫn chiếu từ **top-5** unit thay vì top-50.

## E4 — Nhúng kèm tiêu đề Điều (A/B trên rổ dense thật)

Kernel `zakhim/dsc-legalqa-e4-title-embedding`, private, T4, chạy 21:24→23:01 (97 phút).
Hai nhánh nhúng trong **cùng một phiên**, cùng model `AITeamVN/Vietnamese_Embedding`,
`max_seq_length=1024`, `batch_size=256`, fp16 — đổi đúng một biến là text đem nhúng.

```
Doc xong 432473 unit trong 13.6s          ← khớp E1
  Dieu khong co tieu de        : 1.307
  Dieu co text == dieu_tieu_de : 7.163    ← khớp số patch_layer2_embeddings từng vá
  Hang THUC SU doi text        : 424.003 = 98,0%
Embed bare  2597.3s · Embed title 2887.8s · shape (432473, 1024)
```

432.473 − 424.003 = **8.470** hàng không đổi = đúng 1.307 + 7.163. Mỗi Điều bị chặn đóng góp
đúng một hàng (chúng đều là Điều không chia Khoản) → logic chặn chạy đúng, không sót không thừa.
E4 là can thiệp **rộng: 98% số hàng đổi nội dung nhúng**, không phải chỉnh sửa ở rìa.

### Kết quả, n=1.000 câu `qa_train`, bootstrap cặp 2.000 lần seed 42

| Ngưỡng phủ | BARE | TITLE | Chênh | CI 95% | Đạt? |
|---|---:|---:|---:|---|---|
| ≥70% | 89,4% | 90,3% | +0,90đ | [−0,10, +2,00] | ✗ (chạm trần) |
| ≥80% | 76,4% | **80,8%** | **+4,40đ** | [+3,00, +5,90] | ✓ |
| ≥90% | 61,0% | **65,5%** | **+4,50đ** | [+3,10, +5,90] | ✓ |
| ≥95% | 48,6% | **51,1%** | **+2,50đ** | [+1,50, +3,60] | ✓ |

| | BARE | TITLE | Chênh | CI 95% |
|---|---:|---:|---:|---|
| Phủ chữ trung bình | 0,8812 | 0,8918 | +0,0106 | [+0,0070, +0,0144] |
| METEOR (ngân sách 400) | 0,6326 | **0,6543** | **+0,0217** | [+0,0142, +0,0291] |

125 câu tốt hơn · **75 câu tệ hơn** · 800 không đổi.

### Ba cảnh báo khi dùng con số này

1. **Mức ≥70% không đạt** — cả hai nhánh đã ~90%, chạm trần. E4 **không cứu** câu đang trượt
   hoàn toàn; nó cải thiện thứ hạng trong nhóm đã với tới được.
2. **75 câu tệ đi** (7,5%) — khác hẳn E5 (8 thắng/1 thua, gần như một chiều). Ròng vẫn dương và
   CI loại trừ 0, nhưng đây là nhóm thật. Nếu V9 không thắng LB, soi nhóm này trước.
3. **KHÔNG phải thang của `PLAN_NANG_CAP` Phase 2b.** Ngưỡng "86,4% → ≥88,4%" đo recall@50
   **cấp Khoản trên gold gộp**; ở đây là "top-50 có Điều phủ ≥X% chữ Khối 2", nhãn theo chuỗi
   trên `qa_train`. Cùng hướng, **khác thang — đừng ghép hai con số**.

Phép chấm tra text bằng **row index**, không qua `unit_id`, nên bug `retrieve.py:413` không
chạm vào bảng này. Script tự `assert` 432.473 hàng duyệt lại khớp `unit_index.tsv`.

### Artefact còn trên Kaggle

`corpus_embeddings_bare.npy` và `corpus_embeddings_title.npy` (fp16, ~880 MB mỗi bản) nằm
trong `/kaggle/working/layer2_ab/` của kernel. **Không cần tải về** — phiên Kaggle sau mount
kernel này làm `kernel_sources` là dùng lại được, khỏi nhúng lại 50 phút GPU.

Bản nhẹ đã tải: `outputs/e4/` (20 MB) — top-50 cho 2 nhánh × 3 bộ câu hỏi, kèm `unit_index.tsv`.

---

## Việc còn chặn

| Cần | Vì sao |
|---|---|
| `data123/` | Không có trên máy này, không nằm trong git. Chặn gold gộp 286, `build_citation_labels.py`, P6, P8 |
| `b0_labels_train_sample.json` | `build_b0_labels_train_heldout.py` chết ở dòng 52 nếu thiếu. Rebuild được (seed=42, ~39 phút) |
| `bm25_khoan_index.pkl` | Script build (`build_expensive_indexes.py`) nằm trong `.gitignore`, phải tự viết lại |
| Rổ ứng viên thật | E4 đang sinh trên Kaggle — sẽ mở khoá E3 bản đầy đủ, E5 trên rổ thật |
