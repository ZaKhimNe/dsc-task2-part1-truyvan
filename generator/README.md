# LegalQA Task 2 — Production V8.3 Full Hard

Pipeline end-to-end: JSON retrieval → Qwen3-1.7B + QLoRA adapter →
`answers.json`, `run_metrics.json`, `details.jsonl`.

Module retrieval trong `ChanTaooDe--main/` được nối qua
`scripts/run_retrieval.py`. Xem [thực nghiệm Kaggle](experiments/README.md)
để chạy retrieval → LLM, so sánh base/adapter và đánh giá có kiểm soát.

## Cấu trúc

```text
legalQA_Task2/
├── configs/          Cấu hình model, config local và config mẫu
├── src/              Logic inference, prompt, xử lý input và chấm điểm
├── scripts/          Runner, tải adapter và dữ liệu NLTK
├── kaggle/           Chạy pipeline production trên Kaggle
├── tests/            Kiểm tra pipeline và logic inference
├── inputs/           JSON retrieval, chỉ ví dụ được đưa lên Git
├── models/adapter/   Adapter local, Git bỏ qua
├── .cache/           Base model cache, Git bỏ qua
├── outputs/          Kết quả từng lần chạy, Git bỏ qua
├── requirements.txt
├── version.json
└── run_local.ps1
```

Không cần bộ dữ liệu training hoặc code thực nghiệm để chạy production.
Hai package trong `src/` giữ tên imports cũ và cùng phục vụ một pipeline.

## Thiết lập trên máy mới

Chọn và kích hoạt môi trường Python phù hợp với máy của bạn, rồi chạy lệnh
từ thư mục `legalQA_Task2`. Các lệnh `python` dưới đây dùng Python của môi
trường đang hoạt động, không yêu cầu tên hoặc vị trí môi trường cụ thể.

```powershell
python -m pip install -r requirements.txt
python scripts/download_nltk_data.py
Copy-Item configs\production.example.json configs\production.json
Copy-Item kaggle\settings.example.json kaggle\settings.json
```

Chỉ copy config mẫu khi chưa có config cá nhân. Máy hiện tại đã có config và
adapter, không cần ghi đè. Cài PyTorch phù hợp GPU nếu muốn chạy CUDA; bản CPU
vẫn chạy được nhưng chậm hơn.

## Chạy với adapter có sẵn
Giữ `adapter.local_path` trong `configs/production.json` là `models/adapter`.
Sau khi cài dependencies, chỉ cần chọn input và chạy theo mục tiếp theo.

Inference dùng cả adapter và base model `Qwen/Qwen3-1.7B`. Base model được
lưu trong `.cache/huggingface/` và chỉ tải từ Hugging Face khi cache chưa có.
Máy hiện tại đã có cả base model cache và adapter.

### Tải base model khi chưa có cache

File tự tải base model là **`scripts/run_inference.py`**. Khi chuyển sang máy
mới hoặc đã xóa cache, sau khi cài dependencies và chuẩn bị adapter/input trong
`configs/production.json`, chạy từ thư mục `legalQA_Task2`:

```powershell
python scripts/run_inference.py --limit 1
```

Runner tự tải tokenizer và trọng số
`Qwen/Qwen3-1.7B` từ Hugging Face vào
`legalQA_Task2/.cache/huggingface/`, sau đó nạp adapter và sinh đáp án cho
**1 câu đầu tiên**. Đây là lệnh tải model **kèm chạy thử inference**, không phải
lệnh chỉ tải file. Cần Internet khi tải; các lần chạy sau dùng lại cache đã có.
`scripts/download_adapter.py` chỉ tải adapter, không tải base model cache.

## Chọn input và chạy

Đặt JSON từ thành viên B trực tiếp trong `inputs/`, rồi chọn file trong
`configs/production.json`:

```json
{
  "input_json": "inputs/qa_packages_public_v3_top1_raw.json",
  "top_k": 3,
  "limit": null
}
```

Đây là các trường cần sửa trong config đầy đủ. `top_k` chọn số context đầu
của mỗi câu; `limit: null` chạy tất cả câu, hoặc đặt số nguyên dương để chạy thử.
Đường dẫn trong config tính từ project, còn đường dẫn CLI tính từ thư mục hiện tại.

```powershell
# Thử 1 câu
python scripts/run_inference.py --limit 1
# Chạy toàn bộ input đã chọn
python scripts/run_inference.py
# Chọn input khác
python scripts/run_inference.py --input-json "inputs/retrieval.json" --limit 5
```

Input là danh sách mẫu có `id`, `question`, `contexts`; mỗi context có `text`.
`inputs/example.json` là dữ liệu giả lập minh họa. `reference_answer` là tùy
chọn và chỉ dùng chấm điểm, không đưa vào prompt sinh đáp án.

Mỗi lần chạy tạo `outputs/<run-id>/`:

- `answers.json`: đúng cấu trúc `{"147194": {"answer": "..."}}`.
- `run_metrics.json`: thông tin lần chạy, model, input, số mẫu, thời gian,
  METEOR và ROUGE-L.
- `details.jsonl`: context được chọn và trace generation từng câu.

Khi không có reference, điểm là `null` kèm trạng thái không có đáp án tham chiếu.

## Chạy Kaggle

Sửa `kaggle/settings.json`: username, kernel/dataset slug, `adapter_slug`,
`input_json` (chỉ tên file trong `inputs/`), `top_k`, `limit`.
Adapter đã có trên Kaggle thì gắn dataset bằng `adapter_slug`.

```powershell
.\kaggle\kaggle.ps1 validate
.\kaggle\kaggle.ps1 build-input-dataset
.\kaggle\kaggle.ps1 build-kernel
```

Build chỉ tạo file local trong `kaggle/build/` và thay thế build trước đó.
Bundle chứa source và config runtime, không chứa weights hoặc JSON retrieval.

Tạo private input dataset lần đầu và chạy:

```powershell
.\kaggle\kaggle.ps1 push-input-dataset
.\kaggle\kaggle.ps1 push-kernel
```

Nếu dataset đã tồn tại, đổi input trong settings rồi cập nhật:

```powershell
.\kaggle\kaggle.ps1 update-input-dataset -Message "Update input 2026-09-05"
# Chờ dataset version mới sẵn sàng:
.\kaggle\kaggle.ps1 push-kernel
.\kaggle\kaggle.ps1 status
.\kaggle\kaggle.ps1 logs
.\kaggle\kaggle.ps1 download
```

`update-input-dataset` build input rồi gọi `kaggle datasets version`;
`push-kernel` build và gửi kernel lên Kaggle để chạy. Kết quả tải về
`kaggle/downloads/`. Các lựa chọn input, adapter, top_k, seed, limit trên Kaggle
lấy từ Kaggle settings; lựa chọn generation còn lại lấy từ config local nếu có.

## Kiểm thử và GitHub

```powershell
python -m unittest discover -s tests -v
git status --short
```

Repository được quản lý từ thư mục cha `DSC2026`; không tạo `.git` riêng trong
`legalQA_Task2`. Repo gồm `legalQA_Task2/`, `docs/`, `reports/` và README ngoài.
File `legalQA_Task2/.gitignore` loại
settings cá nhân, credentials, adapter/base model, input thật, cache, build và
outputs. Các config `*.example.json` và input giả lập được đưa lên Git.
Adapter có sẵn trong folder local nhưng hiện không được Git theo dõi vì
`models/` nằm trong `.gitignore`. Vì vậy bản clone từ GitHub không tự có adapter.
Nếu chuyển cả folder project kèm `models/adapter/` sang máy khác thì dùng luôn
adapter đó.

Chỉ khi **thiếu folder adapter** (ví dụ clone riêng code từ GitHub), hãy copy
adapter có sẵn vào `models/adapter/`. Hoặc điền `adapter.kaggle_slug` trong
`configs/production.json`, đăng nhập Kaggle CLI và chạy lệnh tải bổ sung:

```powershell
python scripts/download_adapter.py
```

Script tải chỉ dùng để bổ sung file còn thiếu; không ghi đè adapter đã tồn tại.

## Dual GPU và batch inference

Mặc định inference tự dùng tối đa **2 GPU**, mỗi GPU một process/model riêng và
**batch 10 câu/GPU** (tối đa 20 câu cùng lúc). Máy chỉ có một GPU hoặc CPU vẫn chạy
được. Input được chia xen kẽ; seed vẫn bằng seed gốc + vị trí câu trong input;
answers/details được ghép về đúng thứ tự. Worker lỗi sẽ làm run thất bại, không
xuất bộ kết quả thiếu câu như thể đã hoàn tất.

```powershell
# Tự nhận GPU, batch 10 trên mỗi GPU (Kaggle cũng dùng mặc định này)
python scripts/run_inference.py --batch-size 10
# Bắt buộc có hai GPU
python scripts/run_inference.py --gpu-count 2 --batch-size 10
# Chạy đối chứng theo luồng tuần tự
python scripts/run_inference.py --gpu-count 1 --batch-size 1
```

Batch áp dụng cho decoding greedy, gồm lượt đầu và các retry có cùng ngân sách
token. Sampling dùng từng câu để giữ seed riêng. CUDA hết VRAM sẽ chia nhỏ batch
và ghi log; nếu một câu riêng vẫn hết VRAM thì run báo lỗi. Không tăng giới hạn
token hay đổi quality gate. Với tính toán GPU, batching có thể làm thay đổi đáp
án dù giữ greedy/seed; cần so sánh độ đầy đủ và chất lượng trước khi thay bản chạy.

`run_metrics.json` ghi `gpu_count`, `worker_count`, `batch_size_per_worker`
(kích thước yêu cầu, có thể giảm khi OOM), `sample_latency_mode` và
`timing_includes_model_loading`. Khi batch, latency từng câu được phân bổ từ
thời gian lượt sinh chung; dùng `elapsed_seconds` toàn run để đo tốc độ thực tế.
Thời gian run mới bao gồm tải model, khác cách tính phiên bản tuần tự cũ.

`kaggle.ps1 push-kernel` tự build lại code mới. Runtime phải được cấp hai GPU
thì log mới ghi `Inference: 2 GPU, 2 worker(s), batch_size=10/worker`;
chỉ bật GPU trong metadata không bảo đảm runtime có hai thiết bị. Chưa có kết
quả benchmark dual GPU trong máy kiểm thử local.
