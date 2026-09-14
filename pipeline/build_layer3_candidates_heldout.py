"""
Sinh candidate top-50/câu (Layer 1 BM25 ∥ Layer 2 dense, union, xếp hạng
dense — `search_units_hybrid`) cho 800 câu TẬP GIỮ KÍN (`data/train.json`,
xem `outputs/b0_labels_train_heldout.json` — Phase A của plan
resilient-snacking-firefly.md), phục vụ chạy Layer 3 (reranker) trên Kaggle
lần 2, để đánh giá Layer 3 ở ĐÚNG CẤP KHOẢN trên dữ liệu chưa từng dùng tune.

KHÁC `pipeline/build_layer3_candidates.py` (bản chạy cho `data_retrieve/
train.json`, 7.000 câu) ở nguồn câu hỏi: đây là 800 câu từ `data/train.json`
(tập QA thật), dùng embedding câu hỏi ĐÃ CÓ SẴN
`outputs/layer2/query_embeddings_qa_train.npy` (đã embed toàn bộ qa_train
trên Kaggle từ trước, không cần tính lại).

Chạy: python pipeline/build_layer3_candidates_heldout.py
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
HELDOUT_LABELS_PATH = config.OUTPUTS_DIR / "b0_labels_train_heldout.json"
TOP_K_DOCS_LAYER1 = 100
DENSE_TOP_M = 300
TOP_K_OUT = 50
OUT_PATH = Path("kaggle_layer3") / "upload" / "layer3_candidates_heldout.jsonl"


def main():
    print("Đang load BM25 index + parsed_corpus + embedding Layer 2...", flush=True)
    t0 = time.time()
    bm25_index = retrieve.load_index(config.OUTPUTS_DIR / "bm25_doc_index.pkl")
    parsed_corpus = io_utils.load_parsed_corpus()
    corpus_emb = np.load(LAYER2_DIR / "corpus_embeddings.npy").astype(np.float32)
    with open(LAYER2_DIR / "corpus_unit_ids.json", encoding="utf-8") as f:
        corpus_unit_ids = json.load(f)
    emb_index = retrieve.build_corpus_embedding_index(corpus_unit_ids)
    q_emb = np.load(LAYER2_DIR / "query_embeddings_qa_train.npy").astype(np.float32)
    with open(LAYER2_DIR / "query_qids_qa_train.json", encoding="utf-8") as f:
        q_qids = json.load(f)
    qid_to_qemb = dict(zip(q_qids, q_emb))
    print(f"  -> {time.time() - t0:.1f}s", flush=True)

    train = io_utils.load_train()
    with open(HELDOUT_LABELS_PATH, encoding="utf-8") as f:
        heldout_ids = json.load(f)["heldout_ids"]

    n_missing_qemb = sum(1 for qid in heldout_ids if qid not in qid_to_qemb)
    print(f"  -> {len(heldout_ids)} câu giữ kín, {n_missing_qemb} thiếu embedding câu hỏi (bỏ qua)",
          flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_written = 0
    n_empty_text = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for qid in heldout_ids:
            if qid not in qid_to_qemb:
                continue
            question = train[qid]["question"]
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
            if n_written % 200 == 0:
                print(f"  ... {n_written}/{len(heldout_ids)}, {time.time() - t0:.1f}s", flush=True)

    print(f"Xong: {n_written} câu, {time.time() - t0:.1f}s -> {OUT_PATH}", flush=True)
    print(f"Kiểm tra: số candidate có text RỖNG = {n_empty_text} (kỳ vọng 0)", flush=True)
    print(f"Kích thước file: {OUT_PATH.stat().st_size / 1024 / 1024:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
