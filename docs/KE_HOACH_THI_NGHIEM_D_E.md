# Kế hoạch thí nghiệm D và E — giao cho người chạy

DSC 2026 · LegalQA (Nội dung 2) · viết 21/09/2026 · hạn nộp private **23/09** (tra giờ chốt chính xác trước khi bấm lượt 6,9 giờ nào)

**Đọc hết mục 0 và mục 1 trước khi viết một dòng code.** Mục 0 là những gì đã biết — để khỏi làm lại. Mục 1 là những bẫy đã dẫm tuần này — mỗi cái tốn từ một giờ đến một ngày.

---

## 0. Bối cảnh trong một trang

### Bài đang nộp và cấu hình sản xuất

| | |
|---|---|
| Private LB hiện tại | **0,5383** |
| Luật chấm (Điều 5) | tối đa 3 lượt/ngày; **lấy bản cao nhất** → nộp thêm không bao giờ làm mất 0,5383 |
| Luật mô hình (Điều 4) | chỉ dùng mô hình đã đăng ký; **không sinh trọng số mới** (fine-tune = mô hình mới, hạn đăng ký đã qua) |

Cấu hình sản xuất **mới nhất** (sau T12, 21/09) — đây là **mốc so** của cả D và E:

```
nhúng dense      AITeamVN/Vietnamese_Embedding, Khoản có ghép tiêu đề Điều (nhánh "title", E4)
rổ               hybrid top-50 (thực chất ≈ dense: trùng dense top-50 tới 96%)
reranker         AITeamVN/Vietnamese_Reranker, ĐỌC "tiêu đề Điều + Khoản"  (T12)
trộn thứ hạng    s = α/(60 + dense_rank) + (1−α)/(60 + rerank_rank),  α = 0,5
đóng gói         MAX_CONTEXTS = 2 · ngân sách 200 âm tiết mỗi đoạn · F2/G1 · không cap tiêu đề
generator        Qwen3-1.7B + adapter QLoRA v8, top_k = 2, seed 2026
```

### Nút thắt đã đo — D và E nhắm vào đâu

| Đo | Kết quả | Nghĩa |
|---|---|---|
| T1 — gold Điều ngoài top-50 | chỉ **9,6–13,7%** | đáp án **đã có mặt** trong rổ ở ~90% số câu |
| T6 — trần nếu xếp hoàn hảo trong rổ | top-10: **+0,0822** · top-50: +0,1159 (thang harness) | **71% dư địa nằm trong top-10** |
| T9 — xáo lại bằng hai tín hiệu cũ (dense, rerank) ở mọi cửa sổ | +0,0019 → kiểm chéo **−0,0002** | tín hiệu cũ **không** chạm được dư địa; cần **tín hiệu mới** |
| T3 — reranker có trần | ~60% hạng 1, kéo nhánh mạnh xuống | reranker hiện tại không phải bộ chấm đủ giỏi |
| T12 — reranker đọc tiêu đề | +0,0121 tại α=0,5, CI [+0,0072; +0,0169] | tín hiệu tốt hơn **có** chuyển thành điểm |

Hệ quả: **D nhắm đúng nút thắt** (thứ tự trong top-10, bằng một tín hiệu mới). **E nhắm vào sự có mặt** — thứ không phải nút thắt — nên kỳ vọng thấp hơn và phải qua một cửa rẻ trước.

### Hướng đã đóng — đừng làm lại

Fine-tune bi-encoder (P2, recall@50 −0,23) · cửa sổ chấm lại top-W (T9) · nới ngân sách token (V5 và T10, dấu âm) · nở đoạn ra cả Điều · đọc dẫn chiếu chéo (E5) · vá lỗ cắt BM25 riêng lẻ · chuẩn hoá NFC.

---

## 1. Bảy luật làm việc — mỗi luật đã trả giá thật

**1.1 Mẫu đánh giá mới, không dùng lại mẫu đã chỉnh tham số.**
Mẫu 1909 (`outputs/t3b_qids.json`) đã dùng chỉnh α, k, ngân sách. Mẫu 2109 (`outputs/t12_qids.json`) đã dùng chọn α=0,5. Dùng **seed 2209**, 1.000 câu, rút từ nhóm câu tách được ba khối (harness cần), **giao với 1909 và 2109 = 0** — `assert` điều đó trong code.

**1.2 Ngưỡng viết ra TRƯỚC khi chạy, không sửa sau khi thấy số.** Mục 2.5 và 3.5 đã viết sẵn. Nếu thấy cần đổi ngưỡng thì ghi rõ "đổi sau khi thấy số" trong báo cáo.

**1.3 Tham số mới chọn trên nửa A, đo trên nửa B, rồi đảo lại.** Báo **cả** con số toàn mẫu **và** con số kiểm chéo, và hiệu của chúng. T9 chết đúng ở đây: toàn mẫu +0,0019, kiểm chéo −0,0002.

**1.4 Nhánh đối chứng phải được chỉnh đối xứng.** Nếu nhánh mới được chọn tham số thì nhánh cũ cũng vậy — hoặc cả hai cố định. Chỉnh một bên là cho nó thêm một nút để thắng.

**1.5 Mỗi con số mang danh tính lượt chạy.** In md5 của file rổ, file gói, `pipeline.py` vào log. Ghi seed, cấu hình, giờ chạy cạnh mọi con số trong báo cáo. Tuần này đã có hai lần một con số của lượt cũ bị trích như của lượt mới.

**1.6 Có phép tự kiểm "trọng số 0 phải ra đúng mốc".** Ở T12: α=1,0 cho chênh lệch **đúng 0,0000** vì reranker không được dùng. D và E đều có điểm tương tự (β=0, w=0) — nếu điểm đó không ra đúng mốc tới chữ số cuối thì có lỗi, dừng lại.

**1.7 Kiểm bằng thứ phân biệt được đúng với sai.** "Chạy được" không phải bằng chứng. Tuần này `\n` trong chuỗi bị nuốt một tầng escape ở năm chỗ, ba chỗ vẫn chạy được và cho kết quả sai âm thầm. Cái bắt được là `assert` so dấu phân cách với `chr(92)+chr(110)`.

### Mẹo hạ tầng đã học

- Kaggle CLI: mọi lệnh `datasets create/version` chạy bằng **đường tương đối**. Slug kernel lấy từ **tiêu đề**, không phải trường `id`; lệch thì lần đẩy hai trả **409**.
- Dataset và kernel đều **Private** — dữ liệu BTC không tái phát tán. Sau khi tạo, **kiểm lại** là private, đừng tin mặc định.
- `pickle.load` chỉ mục BM25 ở **ô lệnh đầu tiên** (nó lưu đường module `src.b2_retrieval.retrieve.BM25Index`).
- Chạy thử **50 câu** đo tốc độ thật rồi mới ngoại suy.
- Harness: dùng lại hàm chấm trong `pipeline/eval_t12_rerank_title.py` — đã kiểm trùng khít bộ chấm BTC (`meteor_score([ref.split()], hyp.split())`, tham số mặc định).

---

## 2. Thí nghiệm D — Qwen3-1.7B làm giám khảo cho top-10

### 2.1 Ý tưởng

Hỏi model: *Khoản này có trả lời được câu hỏi không?* Lấy xác suất của câu trả lời "có" làm điểm. Dùng điểm đó như **tín hiệu xếp hạng thứ ba**, trộn với thứ tự sản xuất trong top-10.

Cơ sở: RankGPT (Sun và cộng sự, 2023, *"Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents"*). Bài đó dùng LLM lớn và xếp hạng theo danh sách; ở đây dùng model nhỏ và **chấm từng cặp** (pointwise) — đơn giản hơn, song song được, không cần prompt dài.

**Kỳ vọng thấp, nói trước.** Model 1,7B nhỏ. Nhưng nó là tín hiệu **mới** thật sự — không phải biến đổi của dense hay reranker cũ — và T9 chứng minh ta đang thiếu đúng thứ đó.

### 2.2 Điều 4

Qwen3-1.7B là mô hình đã đăng ký và đang dùng trong bài nộp. Không huấn luyện gì. Dùng **model gốc** `Qwen/Qwen3-1.7B`, **không** gắn adapter QLoRA — adapter được dạy để viết câu trả lời ba khối, không phải để phán đúng/sai.

Việc duy nhất phải làm: **đọc lại bản đăng ký mô hình của nhóm** xem có ghi giới hạn mục đích sử dụng không. Nếu có thì dừng.

### 2.3 Cách làm

**Bước 1 — rổ cho mẫu 2209.** Chạy kernel `kaggle_layer3/rerank_title_input_notebook.py` (kernel T12) với `QIDS` = mẫu 2209. Ra rổ top-50 có `dense_rank` và điểm reranker-tiêu-đề. Khoảng 25–40 phút GPU.

**Bước 2 — thứ tự sản xuất.** Trộn α=0,5 như sản xuất, lấy **top-10** của mỗi câu. Đây là thứ D được phép xáo.

**Bước 3 — chấm bằng Qwen.** Mỗi cặp (câu hỏi, ứng viên) thành một prompt. Ứng viên đưa vào dạng **tiêu đề Điều + Khoản**, qua **đúng hàm tạo chữ của kernel T12** — không viết lại hàm, vì viết hai lần là hai chỗ lệch nhau.

Mẫu prompt (tiếng Việt, một lượt chat):

```
Câu hỏi: {question}

Đoạn luật:
{tieu_de}
{khoan}

Đoạn luật trên có chứa thông tin trả lời trực tiếp câu hỏi không? Chỉ trả lời "Có" hoặc "Không".
```

Lấy điểm ở **token đầu tiên của câu trả lời**, một lần forward, không sinh:

```
điểm = logit("Có") − logit("Không")      (hoặc log-softmax trên đúng hai token đó)
```

Ba bẫy kỹ thuật phải chặn bằng `assert`:

- **Qwen3 có chế độ suy nghĩ.** Phải tắt (`enable_thinking=False` khi dựng chat template). Không tắt thì token đầu là `<think>` và điểm vô nghĩa. Kiểm: in 3 prompt đã dựng, xem cuối prompt đúng là chỗ model bắt đầu trả lời.
- **"Có" / "Không" có thể bị tách thành nhiều token.** Lấy id token đầu của mỗi nhãn, `assert` hai id khác nhau. Nếu tokenizer tách xấu thì đổi nhãn sang cặp có token đầu đơn và khác nhau (ví dụ "A"/"B" kèm hướng dẫn), và ghi rõ đã đổi.
- **Cắt độ dài.** Đặt giới hạn đầu vào (ví dụ 1.024 token), cắt **thân Khoản** chứ không cắt câu hỏi hay phần hướng dẫn. Đếm và in số prompt bị cắt.

Khối lượng: 1.000 câu × 10 ứng viên = **10.000 prompt**, chỉ prefill. Chạy thử 50 câu (500 prompt) để đo tốc độ trước.

**Bước 4 — trộn.** Trong top-10, xếp lại theo:

```
s = (1−β) / (60 + hạng_sản_xuất)  +  β / (60 + hạng_Qwen)
```

Ngoài top-10 giữ nguyên thứ tự sản xuất. Lưới β = {0; 0,2; 0,4; 0,6; 0,8; 1,0}.

**β = 0 phải cho METEOR đúng bằng mốc tới chữ số cuối.** Không bằng thì có lỗi.

**Bước 5 — đóng gói và chấm.** Build bằng `pipeline/build_qa_packages_public_v8_tieu_de_khoan.py` với `--questions train --max-contexts 2 --per-item 200`, rồi chấm harness. Để build nhận thứ tự mới, xuất một file rổ đã sắp lại theo đúng định dạng `ranked_units`, rồi chạy build với `--alpha 0` (giữ nguyên thứ tự file). Kiểm: với β=0, file đó phải cho đúng gói của mốc (so md5 gói).

### 2.4 Chẩn đoán phụ, không quyết định gì

**AUC của điểm Qwen:** trong top-10, điểm của ứng viên chứa Điều gold (nhãn theo chuỗi, như E3a/E5) có cao hơn ứng viên khác không. AUC ≈ 0,5 nghĩa là Qwen không có tín hiệu — khi đó không cần đọc METEOR, đóng luôn. So cùng AUC với điểm reranker-tiêu-đề để biết Qwen có hơn không.

### 2.5 Ngưỡng — đặt trước, cả ba phải đạt

```
1  METEOR harness tại β chọn trên nửa A, đo trên nửa B  ≥ +0,010 so với mốc (β=0)
2  cận dưới CI95 bootstrap ghép cặp (2.000 lượt, seed 42)  > 0
3  đảo hai nửa: β chọn trên B, đo trên A — cũng phải dương
```

Ngưỡng đặt ở **β chọn lại** chứ không ở một β cố định, vì một tín hiệu mới chắc chắn làm dịch trọng số tối ưu — đó là lỗi thiết kế đã mắc ở T12 (xem báo cáo phương pháp §2.7). n=1.000 cho CI khoảng ±0,005 trên METEOR ghép cặp, nên ngưỡng +0,010 phân giải được.

### 2.6 Chi phí và triển khai nếu thắng

| Việc | Thời gian |
|---|---|
| rổ mẫu 2209 | 25–40 phút GPU |
| Qwen chấm 10.000 prompt | ước 10–30 phút GPU — **đo bằng 50 câu trước** |
| build + harness 6 giá trị β | vài phút CPU |
| **nếu thắng:** Qwen chấm 1.918 × 10 prompt private | ~2× lượt train |
| **nếu thắng:** sinh 1.918 câu | ~2 giờ 11 phút (gom lô hai GPU) |

---

## 3. Thí nghiệm E — HyDE

### 3.1 Ý tưởng

Cho Qwen viết một **đoạn luật giả** có thể trả lời câu hỏi, rồi nhúng đoạn đó (hoặc trộn với câu hỏi) để tìm, thay vì chỉ nhúng câu hỏi. Nó bắc cầu giữa câu hỏi đời thường khoảng 19 âm tiết và văn luật trang trọng.

Cơ sở: HyDE (Gao và cộng sự, 2022, *"Precise Zero-Shot Dense Retrieval without Relevance Labels"*).

**Vì sao xếp sau D:** HyDE chủ yếu cứu **sự có mặt** — đưa Điều đúng vào rổ. Mà sự có mặt không phải nút thắt: Điều gold đã nằm trong top-50 ở 86–90% số câu. Dư địa tối đa của E trên recall@50 bị chặn ở khoảng 10–14 điểm, và phần lớn số đó sẽ không lấy được. Nên E có **một cửa rẻ** trước khi được tốn GPU cho cả đường ống.

### 3.2 Điều 4

Qwen3-1.7B và `AITeamVN/Vietnamese_Embedding` đều đã đăng ký. Không huấn luyện. Cùng việc phải làm như D: đọc lại bản đăng ký xem có giới hạn mục đích sử dụng không.

### 3.3 Cách làm — hai tầng

**Tầng 1 (rẻ): chỉ đo sự có mặt, dense thuần.**

Sinh đoạn giả cho 1.000 câu mẫu 2209. Model gốc, tắt suy nghĩ, giải mã tham lam (không lấy mẫu — để chạy lại ra đúng như cũ), `max_new_tokens` khoảng 150–200.

Prompt yêu cầu viết **đúng hình dạng thứ được nhúng trong kho**: vì kho nhánh "title" nhúng *"tiêu đề Điều + nội dung Khoản"*, đoạn giả cũng nên có dạng *một dòng tiêu đề Điều, rồi một Khoản*:

```
Viết một trích đoạn văn bản pháp luật Việt Nam có thể trả lời câu hỏi dưới đây.
Định dạng: dòng đầu là tên Điều, các dòng sau là nội dung một Khoản. Không giải thích.

Câu hỏi: {question}
```

Nhúng đoạn giả bằng **đúng cấu hình đã nhúng kho**:

```
normalize_embeddings = True    ← BẮT BUỘC khớp; lệch là thứ hạng sai hoàn toàn, không báo lỗi
fp16 (model.half())            ← nên khớp
max_seq_length = 1024          ← khớp E4
```

Dùng ma trận nhúng kho **nhánh title** có sẵn trong output kernel E4 (`dsc-legalqa-e4-title-embedding`) — không nhúng lại kho.

Vector truy vấn, lưới w = {0; 0,25; 0,5; 0,75; 1,0}:

```
v = chuẩn_hoá( (1−w) · nhúng(câu hỏi) + w · nhúng(đoạn giả) )
```

**w = 0 phải cho thứ hạng dense đúng bằng mốc.** So danh sách top-50 từng câu, phải trùng 1.000/1.000.

Đo trên cấp **Điều**, nhãn theo chuỗi (như T1): recall@10, recall@50, và hạng trung vị của Điều gold.

**Cửa tầng 1 — đặt trước:**

```
recall@50 cấp Điều tại w chọn trên nửa A, đo trên nửa B  ≥ +2,0 điểm
cận dưới CI95 > 0
```

n=1.000 vượt sàn ≈830 đã tính ở báo cáo phương pháp §2.8 cho ngưỡng ±2 điểm, nên lần này đo phân giải được — khác P2 (n=439).

**Không qua cửa tầng 1 thì đóng E, không làm tầng 2.**

**Tầng 2 (chỉ khi qua cửa): cả đường ống.**

Với **một** giá trị w đã chọn: dựng rổ mới (dense top-300 bằng vector HyDE → union → reranker-tiêu-đề) → trộn α=0,5 → build k=2, ngân sách 200 → harness METEOR. So với mốc sản xuất trên cùng mẫu. Ngưỡng giống D mục 2.5 (+0,010, CI>0, thắng nửa giữ riêng).

### 3.4 Rủi ro đã biết

- Model 1,7B sẽ **bịa** số Điều, số văn bản, mức phạt. Với nhúng thì phần bịa đó chỉ là nhiễu, không phải lỗi — nhưng ghi lại tỉ lệ đoạn giả chứa số hiệu văn bản, vì số hiệu bịa có thể kéo sai văn bản.
- Đoạn giả có thể lặp lại câu hỏi thay vì viết văn luật. In 10 ví dụ ra đọc bằng mắt trước khi chạy đủ 1.000.

### 3.5 Chi phí

| Việc | Thời gian |
|---|---|
| sinh 1.000 đoạn giả | ước 15–30 phút GPU — **đo bằng 50 câu trước** |
| nhúng 1.000 đoạn + tìm trên 432.473 vector | vài phút |
| tầng 2 (nếu qua cửa) | rổ mới 25–40 phút + harness |
| triển khai private (nếu thắng cả hai tầng) | sinh 1.918 đoạn giả + rổ mới + sinh câu trả lời ~2 giờ 11 phút |

---

## 4. Thứ tự và phối hợp

```
song song   D bước 1 (rổ mẫu 2209) — dùng chung cho cả D và E
            E tầng 1 sinh đoạn giả — GPU khác nếu có
tiếp        D bước 3–5 · E đo cửa tầng 1
sau đó      chỉ thí nghiệm nào qua ngưỡng mới được tốn lượt private
```

**Nếu cả D và E cùng thắng:** không cộng dồn mặc định. Chỉ nộp bản gộp khi **bản gộp cũng thắng trên nửa giữ riêng** — hai cái thắng riêng chưa chắc cộng được (T9 là ví dụ: một biện pháp làm lại đúng việc của biện pháp khác).

**Lượt private phải xong trước giờ chốt với biên dư.** Sinh 1.918 câu mất khoảng 2 giờ 11 phút khi gom lô hai GPU; cộng rổ, cộng hàng đợi Kaggle. Bắt đầu muộn nhất khoảng 8 giờ trước giờ chốt.

**Chạy thử 20 câu private trước khi chạy đủ 1.918:** kiểm `selected_context_count` = 2 và `unit_id` đoạn đầu khớp gói. Private không có đáp án nên lỗi nạp dữ liệu không có chỗ nào báo.

---

## 5. Báo cáo kết quả — mẫu phải điền

Mỗi thí nghiệm, gửi lại đúng bảng này:

```
Thí nghiệm           D | E-tầng-1 | E-tầng-2
Mẫu                  seed 2209, n = ____, giao 1909 = 0, giao 2109 = 0  (assert đã chạy: có/không)
Mốc                  cấu hình sản xuất mục 0, METEOR harness = ______
Tự kiểm              β=0 / w=0 trùng mốc: có/không (ghi con số)
Tham số chọn         nửa A → ____ ; nửa B → ____
Toàn mẫu             Δ = ______  CI95 [____; ____]
Kiểm chéo A→B        Δ = ______
Kiểm chéo B→A        Δ = ______
Thiên lệch chọn      toàn mẫu − trung bình kiểm chéo = ______
Ngưỡng               1 đạt/trượt · 2 đạt/trượt · 3 đạt/trượt
Chẩn đoán            D: AUC Qwen ___ vs AUC reranker ___ · E: recall@10/@50, hạng trung vị
md5                  rổ ______ · gói ______ · pipeline.py ______
Thời gian GPU thật   ______
Quyết định           triển khai / đóng — và nếu triển khai dù trượt ngưỡng thì ghi rõ
                     "phán đoán sau khi thấy số", kèm lý do
```

Trượt ngưỡng là một kết quả hợp lệ và có giá trị ngang thắng. Ghi nguyên vẹn, đừng tìm chỉ số phụ đẹp nhất để cứu nó — P2 có recall@5 +2,96 điểm trông hấp dẫn nhưng là một trong bốn chỉ số, và chọn cái đẹp nhất trong bốn chính là thiên lệch chọn.

---

## 6. File và tài nguyên

| Thứ | Ở đâu |
|---|---|
| Báo cáo cho phản biện | `BAO_CAO_PHUONG_PHAP_VA_KET_QUA.md` |
| Nhật ký chi tiết | `BAO_CAO_TONG_HOP_THI_NGHIEM.md` |
| Kernel rổ + reranker-tiêu-đề (train) | `QA/part1-truy-xuat/retrieval/kaggle_layer3/rerank_title_input_notebook.py` |
| Kernel rổ + reranker-tiêu-đề (private) | `QA/part1-truy-xuat/retrieval/kaggle_layer3/rerank_private_title_input_notebook.py` |
| Chấm harness + bootstrap + kiểm chéo | `QA/part1-truy-xuat/retrieval/pipeline/eval_t12_rerank_title.py` |
| Build gói | `QA/part1-truy-xuat/retrieval/pipeline/build_qa_packages_public_v8_tieu_de_khoan.py` |
| Mẫu đã dùng — không dùng lại | `outputs/t3b_qids.json` (1909) · `outputs/t12_qids.json` (2109) |
| Dataset Kaggle | `zakhim/task2-part1` (corpus) · `zakhim/dsc-legalqa-bm25-index` · output kernel `dsc-legalqa-e4-title-embedding` (ma trận nhúng) · `zakhim/dsc2026-legalqa-qlora-v8-adapter` |
| Thể lệ | Điều 4 (mô hình), Điều 5 (nộp bài), Điều 6–7 (công khai mã, bài báo) |
