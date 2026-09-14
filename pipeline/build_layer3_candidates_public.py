"""
Sinh candidate top-50/câu (Layer 1 BM25 ∥ Layer 2 dense, union, xếp hạng
dense — `search_units_hybrid`) cho 1.000 câu TẬP CHÍNH THỨC (`data/public-official.json`),
phục vụ chạy Layer 3 (reranker) trên Kaggle để đánh giá cấp Khoản.

KHÁC `pipeline/build_layer3_candidates_heldout.py` (800 câu tập giữ kín) ở:
- Nguồn: toàn bộ public-official.json (không lọc subset)
- Query embedding: `query_embeddings_qa_public.npy` (đã embed sẵn)

Chạy: python pipeline/build_layer3_candidates_public.py
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
OUT_PATH = Path("kaggle_layer3") / "upload" / "layer3_candidates_public.jsonl"


def main():
    print("Dang load BM25 index + parsed_corpus + embedding Layer 2...", flush=True)
    t0 = time.time()
    bm25_index = retrieve.load_index(config.OUTPUTS_DIR / "bm25_doc_index.pkl")
    parsed_corpus = io_utils.load_parsed_corpus()
    corpus_emb = np.load(LAYER2_DIR / "corpus_embeddings.npy").astype(np.float32)
    with open(LAYER2_DIR / "corpus_unit_ids.json", encoding="utf-8") as f:
        corpus_unit_ids = json.load(f)
    emb_index = retrieve.build_corpus_embedding_index(corpus_unit_ids)
    q_emb = np.load(LAYER2_DIR / "query_embeddings_qa_public.npy").astype(np.float32)
    with open(LAYER2_DIR / "query_qids_qa_public.json", encoding="utf-8") as f:
        q_qids = json.load(f)
    qid_to_qemb = dict(zip(q_qids, q_emb))
    print(f"  -> {time.time() - t0:.1f}s", flush=True)

    qa_public = io_utils.load_public_official()
    public_ids = list(qa_public.keys())

    n_missing_qemb = sum(1 for qid in public_ids if qid not in qid_to_qemb)
    print(f"  -> {len(public_ids)} cau public-official, {n_missing_qemb} thieu embedding (bo qua)",
          flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_written = 0
    n_empty_text = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for qid in public_ids:
            if qid not in qid_to_qemb:
                continue
            question = qa_public[qid]["question"]
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
                print(f"  ... {n_written}/{len(public_ids)}, {time.time() - t0:.1f}s", flush=True)

    print(f"Xong: {n_written} cau, {time.time() - t0:.1f}s -> {OUT_PATH}", flush=True)
    print(f"Kiem tra: so candidate co text RONG = {n_empty_text} (ky vong 0)", flush=True)
    print(f"Kich thuoc file: {OUT_PATH.stat().st_size / 1024 / 1024:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
