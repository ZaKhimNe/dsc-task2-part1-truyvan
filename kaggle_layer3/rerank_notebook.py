# Copy từng đoạn "# %% CELL n" vào 1 cell riêng trong Kaggle Notebook, theo
# đúng thứ tự. Xem README.md (cùng thư mục) để biết cách setup.

# %% CELL 1 — cài thư viện (kèm fix sympy đã gặp ở Layer 2)
!pip install -q -U sympy sentence-transformers

# %% CELL 2 — import + cấu hình đường dẫn
# SỬA tên dataset nếu khác — kiểm tra bằng `!find /kaggle/input/ -maxdepth 4`
# nếu gặp FileNotFoundError (1 số tài khoản Kaggle mount kèm tiền tố
# "datasets/<username>/" — xem kaggle_layer2/README.md mục Xử lý lỗi)
import json
import time
from pathlib import Path
import torch
from sentence_transformers import CrossEncoder

INPUT_DIR = Path("/kaggle/input/datasets/ngmaidnghi/dsc-legalqa-b2-layer3-input")  # xác nhận thật 17/08 (path có tiền tố datasets/<username>/)
OUT_DIR = Path("/kaggle/working/layer3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# %% CELL 3 — load model reranker (~568M tham số)
MODEL_NAME = "BAAI/bge-reranker-v2-m3"
model = CrossEncoder(MODEL_NAME, max_length=512, device=DEVICE)
print("Model loaded.")

# %% CELL 4 — đọc candidate đã sinh sẵn (local, search_units_hybrid — Layer1 BM25
# ∥ Layer2 dense, union cấp unit, top-50 không ràng buộc đa dạng — xem
# pipeline/build_layer3_candidates.py), GỘP TẤT CẢ cặp (câu hỏi, candidate)
# thành 1 batch lớn duy nhất để tận dụng GPU (tránh gọi predict() nhiều lần
# với batch nhỏ — phí sức mạnh song song của GPU).
qids_flat = []
unit_ids_flat = []
context_ids_flat = []
pairs = []

t0 = time.time()
with open(INPUT_DIR / "layer3_candidates_train.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        qid = r["qid"]
        question = r["question"]
        for c in r["candidates"]:
            qids_flat.append(qid)
            unit_ids_flat.append(c["unit_id"])
            context_ids_flat.append(c["context_id"])
            pairs.append((question, c["text"]))
print(f"Đọc xong {len(pairs)} cặp (câu hỏi, candidate) trong {time.time()-t0:.1f}s", flush=True)

# %% CELL 5 — chạy reranker trên TOÀN BỘ cặp cùng lúc (batch lớn, GPU tự chia nhỏ nội bộ)
t0 = time.time()
scores = model.predict(pairs, batch_size=128, show_progress_bar=True)
print(f"Rerank xong {len(pairs)} cặp: {time.time()-t0:.1f}s", flush=True)

# %% CELL 6 — nhóm lại theo câu hỏi, xếp hạng CẤP UNIT (KHÔNG collapse về văn
# bản — bài học từ B4 17/08: đo/rank ở cấp văn bản đánh lừa, cái quan trọng
# là thứ tự đúng ở cấp Khoản). Giữ nguyên toàn bộ candidate đã rerank kèm
# score để B3/B4 tự đánh giá Hit-Khoản, không cắt cứng tại đây.
from collections import defaultdict

per_query: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
for qid, uid, cid, score in zip(qids_flat, unit_ids_flat, context_ids_flat, scores):
    per_query[qid].append((uid, cid, float(score)))

out_path = OUT_DIR / "L3_rerank_train.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for qid, items in per_query.items():
        items.sort(key=lambda x: x[2], reverse=True)
        ranked_units = [{"unit_id": uid, "context_id": cid, "score": score} for uid, cid, score in items]
        f.write(json.dumps({"qid": qid, "ranked_units": ranked_units}, ensure_ascii=False) + "\n")

print(f"Đã lưu {out_path} ({len(per_query)} câu hỏi, cấp unit, không collapse văn bản)")

# %% CELL 7 — nén để tải về (RẤT NHẸ, chỉ vài MB — không phải embedding nặng như Layer 2)
import shutil
shutil.make_archive("/kaggle/working/layer3_results", "zip", OUT_DIR)
print("Đã tạo /kaggle/working/layer3_results.zip — tải file này về (panel Output bên phải).")
