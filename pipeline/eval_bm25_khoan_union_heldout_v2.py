"""
Do lai Phase 3 (BM25-Khoan lam nguon UNION) tren bo nhan GOP (B0 + data123,
n=286) thay vi chi 198 cau B0 (da phat hien thien vi ve phia cau de truy hoi).

Tinh CA HAI (co/khong BM25-Khoan) trong CUNG 1 vong lap de bootstrap CI dung
cap (paired), tranh dung lai loi da gap o lan do truoc (2 quan the khac nhau).

Chay: python pipeline/eval_bm25_khoan_union_heldout_v2.py
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config, io_utils
from src.b2_retrieval import retrieve

NEW_LAYER2_DIR = config.REPO_ROOT / "layer2_embeddings"
MERGED_GOLD_PATH = config.OUTPUTS_DIR / "gold_labels_merged_heldout.json"
TOP_K_DOCS_LAYER1 = 100
DENSE_TOP_M = 300
BM25_KHOAN_TOP_K = 300
TOP_K_OUT = 50


def infer_unit_type(unit_id: str, context_id: str) -> str:
    suffix = unit_id[len(context_id) + 1:] if unit_id.startswith(context_id + "_") else ""
    n_parts = len([p for p in suffix.split("_") if p])
    if n_parts >= 2:
        return "khoan"
    if n_parts == 1:
        return "dieu_fallback"
    return "doc_fallback"


def bm25_khoan_to_units(scored, parsed_corpus):
    units = []
    for s in scored:
        unit_id = str(s["context_id"])
        real_context_id = unit_id.split("_", 1)[0]
        text = retrieve.get_unit_text(parsed_corpus, unit_id)
        units.append({
            "unit_id": unit_id, "context_id": real_context_id, "text": text,
            "doc_score": s["score"], "score": s["score"],
            "unit_type": infer_unit_type(unit_id, real_context_id),
        })
    return units


def get_baseline_and_union_top50(bm25_doc_index, bm25_khoan_index, question, parsed_corpus,
                                  corpus_emb, corpus_unit_ids, emb_index, query_embedding):
    doc_results = retrieve.search_docs_multi(bm25_doc_index, question, top_k_docs=TOP_K_DOCS_LAYER1, use_multi_query=False)
    units_lexical = retrieve.expand_to_units(doc_results, parsed_corpus)
    units_dense = retrieve.dense_search_units(corpus_emb, corpus_unit_ids, query_embedding, DENSE_TOP_M)

    baseline_merged = retrieve.union_units(units_lexical, units_dense)
    baseline_ranked = retrieve.rerank_units_dense(corpus_emb, emb_index, query_embedding, baseline_merged)
    baseline_top50 = {u["unit_id"] for u in baseline_ranked[:TOP_K_OUT]}

    khoan_scored = retrieve.search(bm25_khoan_index, question, top_k=BM25_KHOAN_TOP_K)
    units_bm25_khoan = bm25_khoan_to_units(khoan_scored, parsed_corpus)
    union_merged = retrieve.union_units(units_lexical, units_dense, units_bm25_khoan)
    union_ranked = retrieve.rerank_units_dense(corpus_emb, emb_index, query_embedding, union_merged)
    union_top50 = {u["unit_id"] for u in union_ranked[:TOP_K_OUT]}

    return baseline_top50, union_top50


def bootstrap_ci_diff(hits_a, hits_b, n_boot=2000, seed=42):
    n = len(hits_a)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        a = sum(hits_a[i] for i in idx) / n
        b = sum(hits_b[i] for i in idx) / n
        diffs.append(b - a)
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]


def main():
    print("Dang load BM25 doc + BM25 khoan + parsed_corpus + embedding Layer 2 moi...", flush=True)
    t0 = time.time()
    bm25_doc_index = retrieve.load_index(config.OUTPUTS_DIR / "bm25_doc_index.pkl")
    bm25_khoan_index = retrieve.load_index(config.OUTPUTS_DIR / "bm25_khoan_index.pkl")
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
    merged = json.loads(MERGED_GOLD_PATH.read_text(encoding="utf-8"))
    gold = {qid: set(v["khoan_ids"]) for qid, v in merged.items() if v["source"] != "disagree"}
    print(f"So cau trong bo nhan gop (da bo disagree): {len(gold)}", flush=True)

    t0 = time.time()
    hits_baseline = []
    hits_union = []
    n_scored = 0
    for qid, gold_ids in gold.items():
        if qid not in qid_to_qemb:
            continue
        question = train[qid]["question"]
        baseline_top50, union_top50 = get_baseline_and_union_top50(
            bm25_doc_index, bm25_khoan_index, question, parsed_corpus,
            corpus_emb, corpus_unit_ids, emb_index, qid_to_qemb[qid],
        )
        hits_baseline.append(1 if gold_ids & baseline_top50 else 0)
        hits_union.append(1 if gold_ids & union_top50 else 0)
        n_scored += 1
        if n_scored % 50 == 0:
            print(f"  ... {n_scored}/{len(gold)}, {time.time() - t0:.1f}s", flush=True)

    r_baseline = sum(hits_baseline) / n_scored
    r_union = sum(hits_union) / n_scored
    lo, hi = bootstrap_ci_diff(hits_baseline, hits_union)
    diff = r_union - r_baseline

    print(f"\nSo cau cham duoc: {n_scored}/{len(gold)}")
    print(f"\n=== KET QUA (bo nhan gop, n={n_scored}) ===")
    print(f"Recall@50 KHONG BM25-Khoan (bi-encoder moi, dung nhu Phase 2b): {r_baseline:.1%}")
    print(f"Recall@50 CO UNION BM25-Khoan:                                 {r_union:.1%}")
    print(f"Chenh lech: {diff:+.1%}  CI=[{lo:+.1%},{hi:+.1%}]  {'***' if (lo>0 or hi<0) else ''}")
    if diff >= 0.02 and lo > 0:
        print("=> DUONG CO Y NGHIA -> MO LAI Phase 3, dua vao production.")
    elif hi < 0:
        print("=> AM CO Y NGHIA -> giu dong.")
    else:
        print("=> Khong the phan biet voi nhieu (CI chua 0) -> chua du can cu de mo lai.")


if __name__ == "__main__":
    main()
