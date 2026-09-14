"""
Sinh sẵn candidate/câu hỏi theo THIẾT KẾ CUỐI `search_units_hybrid` (chốt
17/08/2026 — xem docs/EXPERIMENT_LOG.md entry [B2] 17/08) cho `data_retrieve/
train.json` (7,000 câu — KHÔNG dùng warmup.json, đã qua giai đoạn warm-up của
cuộc thi, theo quyết định người dùng 13/08/2026).

BẢN CŨ (trước 17/08) dùng Layer 1 BM25 + Expand + Layer 1.5 rerank BM25,
cắt top-30 — đã LỖI THỜI sau đợt rà soát kiến trúc (xem [B2] 17/08, 6 vấn đề
đã sửa: phễu thắt, union sai cấp, bỏ điểm BM25 khi rerank, max_per_doc phá
Hit-Khoản...). Bản này dùng `search_units_hybrid`:

  Layer 1 (BM25 doc, top_k_docs=100) ∥ Layer 2 (dense unit-level, dense_top_m=300)
  → Union cấp unit → xếp hạng bằng dense score → cắt top_k_out=50, KHÔNG ràng
    buộc đa dạng (max_per_doc=None — ràng buộc này đã đo làm Hit-Khoản sụp
    81.9% -> 43.3%, xem [B4] 17/08)

Cần embedding câu hỏi đã tính sẵn trên Kaggle: query_embeddings_retrieve_train.npy
(khớp data_retrieve/train.json qua query_qids_retrieve_train.json).

Ghi ra file NHẸ (chỉ text cần thiết, không phải cả corpus 700MB) để upload
lên Kaggle — ở đó chỉ cần chạy đúng phần GPU-heavy (reranker Layer 3), không
cần upload lại parsed_corpus.jsonl/BM25 index/embedding corpus.

Chạy: python pipeline/build_layer3_candidates.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config, io_utils
from src.b2_retrieval import retrieve

LAYER2_DIR = config.OUTPUTS_DIR / "layer2"
TOP_K_DOCS_LAYER1 = 100
DENSE_TOP_M = 300
TOP_K_OUT = 50
OUT_PATH = Path("kaggle_layer3") / "upload" / "layer3_candidates_train.jsonl"


def main():
    print("Đang load BM25 index + parsed_corpus + embedding Layer 2...", flush=True)
    t0 = time.time()
    bm25_index = retrieve.load_index(config.OUTPUTS_DIR / "bm25_doc_index.pkl")
    parsed_corpus = io_utils.load_parsed_corpus()
    corpus_emb = np.load(LAYER2_DIR / "corpus_embeddings.npy").astype(np.float32)
    with open(LAYER2_DIR / "corpus_unit_ids.json", encoding="utf-8") as f:
        corpus_unit_ids = json.load(f)
    emb_index = retrieve.build_corpus_embedding_index(corpus_unit_ids)
    q_emb = np.load(LAYER2_DIR / "query_embeddings_retrieve_train.npy").astype(np.float32)
    with open(LAYER2_DIR / "query_qids_retrieve_train.json", encoding="utf-8") as f:
        q_qids = json.load(f)
    qid_to_qemb = dict(zip(q_qids, q_emb))
    print(f"  -> {time.time() - t0:.1f}s", flush=True)

    with open(config.DATA_RETRIEVE_DIR / "train.json", encoding="utf-8") as f:
        train = json.load(f)

    n_missing_qemb = sum(1 for qid in train if qid not in qid_to_qemb)
    print(f"  -> {len(train)} câu hỏi, {n_missing_qemb} thiếu embedding câu hỏi (bỏ qua)",
          flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_written = 0
    n_empty_text = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for i, (qid, item) in enumerate(train.items()):
            if qid not in qid_to_qemb:
                continue
            question = item["question"]
            units = retrieve.search_units_hybrid(
                bm25_index, question, parsed_corpus,
                corpus_emb, corpus_unit_ids, emb_index, qid_to_qemb[qid],
                top_k_docs=TOP_K_DOCS_LAYER1, dense_top_m=DENSE_TOP_M, top_k_out=TOP_K_OUT,
            )
            candidates = [
                {"unit_id": u["unit_id"], "context_id": u["context_id"], "text": u["text"]}
                for u in units
            ]
            n_empty_text += sum(1 for c in candidates if not c["text"].strip())
            f.write(json.dumps({"qid": qid, "question": question, "candidates": candidates},
                                ensure_ascii=False) + "\n")
            n_written += 1
            if n_written % 500 == 0:
                print(f"  ... {n_written}/{len(train)}, {time.time() - t0:.1f}s", flush=True)

    print(f"Xong: {n_written} câu, {time.time() - t0:.1f}s -> {OUT_PATH}", flush=True)
    print(f"Kiểm tra: số candidate có text RỖNG = {n_empty_text} (kỳ vọng 0)", flush=True)
    print(f"Kích thước file: {OUT_PATH.stat().st_size / 1024 / 1024:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
