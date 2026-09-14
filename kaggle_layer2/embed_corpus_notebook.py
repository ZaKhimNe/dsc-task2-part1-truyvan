# Copy từng đoạn "# %% CELL n" vào 1 cell riêng trong Kaggle Notebook, theo
# đúng thứ tự. Xem README.md (cùng thư mục) để biết cách setup dataset +
# GPU + Internet trước khi chạy.

# %% CELL 1 — cài thư viện
# nâng cấp sympy TRƯỚC — image Kaggle sẵn có sympy cũ hơn xung đột với
# torch/transformers mới, gây AttributeError: module 'sympy' has no
# attribute 'core' khi import sentence_transformers (đo thật 13/08/2026,
# lỗi môi trường Kaggle, không phải lỗi code/data của mình). Sau cell này
# BẮT BUỘC Restart Session rồi chạy lại từ đầu — không chỉ chạy lại cell.
!pip install -q -U sympy sentence-transformers

# %% CELL 2 — import + cấu hình đường dẫn
# SỬA "dsc-legalqa-b2-layer2-input" thành đúng tên dataset bạn đặt ở Bước 1
import json
import time
import numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

INPUT_DIR = Path("/kaggle/input/dsc-legalqa-b2-layer2-input")  # SỬA tên dataset nếu khác
# LƯU Ý (đo thật 13/08/2026): 1 số tài khoản Kaggle mount dataset kèm tiền tố
# "datasets/<username>/" thay vì thẳng "/kaggle/input/<dataset-slug>/" — nếu
# gặp FileNotFoundError dù dataset đã add đúng, chạy `!find /kaggle/input/ -maxdepth 4`
# để tìm đường dẫn thật, rồi sửa dòng INPUT_DIR ở trên cho khớp, ví dụ:
# INPUT_DIR = Path("/kaggle/input/datasets/<username>/dsc-legalqa-b2-layer2-input")
OUT_DIR = Path("/kaggle/working/layer2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# %% CELL 3 — load model BGE-M3 (~568M tham số, trong ngân sách <=0.7B)
MODEL_NAME = "BAAI/bge-m3"
model = SentenceTransformer(MODEL_NAME, device=DEVICE)
model.max_seq_length = 1024  # đa số Khoản ngắn (median 222 ký tự ~ 50-70 token) — cap để
                              # tránh unit_type=doc_fallback siêu dài (hiếm, 0.3%) làm chậm cả batch
if DEVICE == "cuda":
    model.half()  # fp16 cho nhanh hơn trên GPU, không đáng kể mất độ chính xác cho embedding
print("Model loaded. Embedding dim:", model.get_sentence_embedding_dimension())

# %% CELL 4 — đọc parsed_corpus.jsonl, trích toàn bộ unit Khoản/Dieu (giống
# logic pipeline/build_expensive_indexes.py build_khoan() ở máy local)
unit_ids: list[str] = []
unit_texts: list[str] = []

t0 = time.time()
with open(INPUT_DIR / "parsed_corpus.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["parse_status"] == "fallback":
            if r["dieu"]:
                d0 = r["dieu"][0]
                unit_ids.append(d0["dieu_id"] or r["context_id"])
                unit_texts.append(d0["text"])
            continue
        for dieu in r["dieu"]:
            if dieu["khoan"]:
                for k in dieu["khoan"]:
                    unit_ids.append(k["khoan_id"])
                    unit_texts.append(k["text"])
            else:
                unit_ids.append(dieu["dieu_id"])
                unit_texts.append(dieu["text"])

print(f"Đọc xong {len(unit_ids)} unit trong {time.time()-t0:.1f}s")
assert len(unit_ids) == len(unit_texts)

# %% CELL 5 — embed toàn bộ corpus (mất lâu nhất, ~20-30 phút trên T4)
t0 = time.time()
corpus_embeddings = model.encode(
    unit_texts,
    batch_size=256,
    show_progress_bar=True,
    normalize_embeddings=True,  # để cosine similarity = dot product, tiện search sau này
    convert_to_numpy=True,
)
print(f"Embed corpus xong: {time.time()-t0:.1f}s, shape={corpus_embeddings.shape}")

np.save(OUT_DIR / "corpus_embeddings.npy", corpus_embeddings.astype(np.float16))  # fp16 giảm size file
with open(OUT_DIR / "corpus_unit_ids.json", "w", encoding="utf-8") as f:
    json.dump(unit_ids, f)
print("Đã lưu corpus_embeddings.npy + corpus_unit_ids.json")

# %% CELL 6 — embed toàn bộ câu hỏi của 5 nguồn (data_retrieve train/warmup,
# QA train/warmup/public) trong CÙNG phiên GPU này, đỡ phải quay lại Kaggle lần 2
QUERY_SOURCES = {
    "retrieve_train": "data_retrieve_train.json",
    "retrieve_warmup": "data_retrieve_warmup.json",
    "qa_train": "qa_train.json",
    "qa_warmup": "qa_warmup.json",
    "qa_public": "qa_public.json",  # data/public-official.json — đích cuối cùng cần predict, tận dụng
                                      # luôn phiên GPU này thay vì phải quay lại Kaggle lần 2
}
# LƯU Ý: đổi tên file khi upload dataset để phân biệt 2 cặp train.json/warmup.json
# trùng tên (data_retrieve/train.json vs data/train.json) — xem README.md Bước 1.
# Nếu bạn giữ nguyên tên gốc (train.json x2, warmup.json x2), sửa dict trên cho khớp
# cấu trúc thư mục thật trong INPUT_DIR (vd INPUT_DIR/"data_retrieve"/"train.json").

for tag, fname in QUERY_SOURCES.items():
    fpath = INPUT_DIR / fname
    if not fpath.exists():
        print(f"BỎ QUA {tag}: không thấy {fpath} — sửa lại đường dẫn trong QUERY_SOURCES nếu cần")
        continue
    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)
    qids = list(data.keys())
    questions = [data[q]["question"] for q in qids]

    t0 = time.time()
    q_emb = model.encode(questions, batch_size=256, show_progress_bar=True,
                          normalize_embeddings=True, convert_to_numpy=True)
    print(f"{tag}: {len(qids)} câu, {time.time()-t0:.1f}s")

    np.save(OUT_DIR / f"query_embeddings_{tag}.npy", q_emb.astype(np.float16))
    with open(OUT_DIR / f"query_qids_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(qids, f)

print("Đã embed xong toàn bộ query set.")

# %% CELL 7 — nén lại để tải về
import shutil
shutil.make_archive("/kaggle/working/layer2_embeddings", "zip", OUT_DIR)
print("Đã tạo /kaggle/working/layer2_embeddings.zip — tải file này về (panel Output bên phải).")

# %% CELL 8 — TÙY CHỌN nếu KHÔNG tải được layer2_embeddings.zip (800MB+ quá nặng,
# hay bị đứt giữa chừng qua trình duyệt): tính luôn bước SEARCH ngay trên Kaggle
# (embedding đang sẵn có trong RAM/GPU ở đây, không cần tải đi đâu), chỉ xuất ra
# KẾT QUẢ XẾP HẠNG (rất nhẹ, vài MB) để tải về thay vì tải nguyên embedding nặng.
import numpy as np

corpus_t = torch.from_numpy(corpus_embeddings if "corpus_embeddings" in dir() else
                             np.load(OUT_DIR / "corpus_embeddings.npy")).to(DEVICE)
if corpus_t.dtype != torch.float16:
    corpus_t = corpus_t.half()
corpus_t = corpus_t.T.contiguous()  # (1024, 394000) — chuyển vị 1 lần, dùng lại nhiều lần

with open(OUT_DIR / "corpus_unit_ids.json", encoding="utf-8") as f:
    corpus_unit_ids = json.load(f)

TOP_K_RAW = 200  # số Khoản thô lấy ra trước khi collapse về văn bản
DOC_TOP_K = 100  # số văn bản riêng biệt giữ lại sau collapse


def collapse_to_doc(top_indices: np.ndarray) -> list[str]:
    """Giống collapse_reranked_to_doc ở máy local (pipeline/build_khoan_doc_run.py)
    — giữ context_id xuất hiện SỚM NHẤT (điểm cao nhất), cắt top DOC_TOP_K."""
    seen: dict[str, None] = {}
    for idx in top_indices:
        cid = corpus_unit_ids[idx].split("_", 1)[0]
        if cid not in seen:
            seen[cid] = None
            if len(seen) >= DOC_TOP_K:
                break
    return list(seen.keys())


def dense_search(tag: str, batch_size: int = 200) -> dict[str, list[str]]:
    q_emb = np.load(OUT_DIR / f"query_embeddings_{tag}.npy")
    with open(OUT_DIR / f"query_qids_{tag}.json", encoding="utf-8") as f:
        qids = json.load(f)

    runs: dict[str, list[str]] = {}
    t0 = time.time()
    for start in range(0, len(qids), batch_size):
        end = min(start + batch_size, len(qids))
        batch_t = torch.from_numpy(q_emb[start:end]).to(DEVICE).half()
        scores_t = batch_t @ corpus_t  # (b, 394000) — nhân ma trận trên GPU, rất nhanh
        scores = scores_t.float().cpu().numpy()
        for i in range(end - start):
            row = scores[i]
            top_idx = np.argpartition(-row, TOP_K_RAW)[:TOP_K_RAW]
            top_idx = top_idx[np.argsort(-row[top_idx])]
            runs[qids[start + i]] = collapse_to_doc(top_idx)
    print(f"{tag}: {len(qids)} câu, {time.time()-t0:.1f}s")
    return runs


DENSE_RUNS_DIR = Path("/kaggle/working/dense_runs")
DENSE_RUNS_DIR.mkdir(exist_ok=True)

for tag in ["retrieve_train", "retrieve_warmup", "qa_train", "qa_warmup", "qa_public"]:
    runs = dense_search(tag)
    with open(DENSE_RUNS_DIR / f"dense_run_{tag}.jsonl", "w", encoding="utf-8") as f:
        for qid, ranked_ids in runs.items():
            f.write(json.dumps({"qid": qid, "ranked_ids": ranked_ids}, ensure_ascii=False) + "\n")

print("Xong toàn bộ dense search — kết quả trong /kaggle/working/dense_runs/")

# %% CELL 9 — nén kết quả search (RẤT NHẸ, vài MB) để tải về thay vì file embedding nặng
shutil.make_archive("/kaggle/working/dense_runs", "zip", DENSE_RUNS_DIR)
print("Đã tạo /kaggle/working/dense_runs.zip — TẢI FILE NÀY (nhẹ), KHÔNG cần tải layer2_embeddings.zip nữa.")
