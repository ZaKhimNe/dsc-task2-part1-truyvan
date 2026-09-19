"""
Sinh lai candidate bi-encoder MOI cho DU 800 cau heldout (khong chi 198 cau
B0 tin cay nhu `eval_layer2_bienconder_heldout.py` truoc do) — de so sanh
cong bang voi file CU (`kaggle_layer3/upload/layer3_candidates_heldout.jsonl`,
da phu du 800 cau tu dau) tren bo nhan gop moi (286 cau, gom ca 97 cau
d123_only ma B0 khong co).

Chay: python pipeline/build_layer3_candidates_heldout_aiteamvn_full.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config, io_utils
from src.b2_retrieval import retrieve

NEW_LAYER2_DIR = config.REPO_ROOT / "layer2_embeddings"
HELDOUT_LABELS_PATH = config.OUTPUTS_DIR / "b0_labels_train_heldout.json"
TOP_K_DOCS_LAYER1 = 100
DENSE_TOP_M = 300
TOP_K_OUT = 50
OUT_PATH = config.OUTPUTS_DIR / "layer3_candidates_heldout_aiteamvn_full.jsonl"


def main():
    print("Dang load BM25 index + parsed_corpus + embedding Layer 2 MOI...", flush=True)
    t0 = time.time()
    bm25_index = retrieve.load_index(config.OUTPUTS_DIR / "bm25_doc_index.pkl")
    parsed_corpus = io_utils.load_parsed_corpus()
    corpus_emb = np.load(NEW_LAYER2_DIR / "corpus_embeddings.npy").astype(np.float32)
    with open(NEW_LAYER2_DIR / "corpus_unit_ids.json", encoding="utf-8") as f:
        corpus_unit_ids = json.load(f)
    emb_index = retrieve.build_corpus_embedding_index(corpus_unit_ids)
    q_emb = np.load(NEW_LAYER2_DIR / "query_embeddings_qa_train.npy").astype(np.float32)
    with open(NEW_LAYER2_DIR / "query_qids_qa_train.json", encoding="utf-8") as f:
        q_qids = json.load(f)
    qid_to_qemb = dict(zip(q_qids, q_emb))
    print(f"  -> {time.time() - t0:.1f}s", flush=True)

    train = io_utils.load_train()
    heldout_ids = json.loads(HELDOUT_LABELS_PATH.read_text(encoding="utf-8"))["heldout_ids"]
    print(f"So cau heldout (DU, khong loc confidence): {len(heldout_ids)}", flush=True)

    t0 = time.time()
    n_written = 0
    n_missing_qemb = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f_out:
        for qid in heldout_ids:
            if qid not in qid_to_qemb:
                n_missing_qemb += 1
                continue
            question = train[qid]["question"]
            units = retrieve.search_units_hybrid(
                bm25_index, question, parsed_corpus,
                corpus_emb, corpus_unit_ids, emb_index, qid_to_qemb[qid],
                top_k_docs=TOP_K_DOCS_LAYER1, dense_top_m=DENSE_TOP_M, top_k_out=TOP_K_OUT,
            )
            candidates = [{"unit_id": u["unit_id"], "context_id": u["context_id"], "text": u["text"]} for u in units]
            f_out.write(json.dumps({"qid": qid, "question": question, "candidates": candidates}, ensure_ascii=False) + "\n")
            n_written += 1
            if n_written % 100 == 0:
                print(f"  ... {n_written}/{len(heldout_ids)}, {time.time() - t0:.1f}s", flush=True)

    print(f"\nXong: {n_written} cau -> {OUT_PATH}", flush=True)
    print(f"Bo qua (thieu embedding cau hoi): {n_missing_qemb}", flush=True)


if __name__ == "__main__":
    main()
