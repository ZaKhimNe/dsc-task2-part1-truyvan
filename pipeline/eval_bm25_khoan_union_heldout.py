"""
Phase 3 (PLAN_KE_THUA_TEAM_IR.md) — thu them nguon BM25 cap Khoan
(`outputs/bm25_khoan_index.pkl`, da build tu truoc, chua tung wire vao
production) lam UNION voi rong dang dung (BM25-doc-expand union dense-moi),
do recall@50 cap Khoan tren 198 cau heldout, so voi moc 90,4% (da dat sau
Phase 2b).

KHONG sua `search_units_hybrid` trong retrieve.py (ham loi, nhieu noi goi) —
viet ham bien the rieng o day, chi dung de thu nghiem.

Nguong dat truoc: >=92,4% (+2 diem%) -> giu; <90,4% -> bo han huong nay.

Chay: python pipeline/eval_bm25_khoan_union_heldout.py
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
CONFIDENCE_TRUST = config.B0_CONFIDENCE_TRUST
TOP_K_DOCS_LAYER1 = 100
DENSE_TOP_M = 300
BM25_KHOAN_TOP_K = 300  # cung thang voi dense_top_m, cho nhanh BM25-Khoan co hoi dong gop cong bang
TOP_K_OUT = 50
BASELINE_RECALL_AT_50 = 0.904  # bi-encoder moi, chua co BM25-Khoan (PLAN_NANG_CAP.md Phase 2b)


def infer_unit_type(unit_id: str, context_id: str) -> str:
    suffix = unit_id[len(context_id) + 1 :] if unit_id.startswith(context_id + "_") else ""
    n_parts = len([p for p in suffix.split("_") if p])
    if n_parts >= 2:
        return "khoan"
    if n_parts == 1:
        return "dieu_fallback"
    return "doc_fallback"


def bm25_khoan_to_units(scored: list[dict], parsed_corpus: dict) -> list:
    units = []
    for s in scored:
        unit_id = str(s["context_id"])  # ten field cu, thuc chat la khoan_id/dieu_id
        real_context_id = unit_id.split("_", 1)[0]
        text = retrieve.get_unit_text(parsed_corpus, unit_id)
        units.append({
            "unit_id": unit_id,
            "context_id": real_context_id,
            "text": text,
            "doc_score": s["score"],
            "score": s["score"],
            "unit_type": infer_unit_type(unit_id, real_context_id),
        })
    return units


def search_units_hybrid_plus_bm25_khoan(
    bm25_doc_index, bm25_khoan_index, question, parsed_corpus,
    corpus_emb, corpus_unit_ids, emb_index, query_embedding,
):
    doc_results = retrieve.search_docs_multi(bm25_doc_index, question, top_k_docs=TOP_K_DOCS_LAYER1, use_multi_query=False)
    units_lexical = retrieve.expand_to_units(doc_results, parsed_corpus)
    units_dense = retrieve.dense_search_units(corpus_emb, corpus_unit_ids, query_embedding, DENSE_TOP_M)
    khoan_scored = retrieve.search(bm25_khoan_index, question, top_k=BM25_KHOAN_TOP_K)
    units_bm25_khoan = bm25_khoan_to_units(khoan_scored, parsed_corpus)

    merged = retrieve.union_units(units_lexical, units_dense, units_bm25_khoan)
    ranked = retrieve.rerank_units_dense(corpus_emb, emb_index, query_embedding, merged)
    out = ranked[:TOP_K_OUT]
    return [u if u["text"] else {**u, "text": retrieve.get_unit_text(parsed_corpus, u["unit_id"])} for u in out]


def load_gold_khoan(labels_path: Path, confidence_trust: float) -> dict[str, set[str]]:
    labels = json.loads(labels_path.read_text(encoding="utf-8"))["labels"]
    gold: dict[str, set[str]] = {}
    for qid, spans in labels.items():
        khoan_ids = {
            s["khoan_id"] for s in spans
            if s["unit_type"] == "khoan" and s["confidence"] >= confidence_trust
        }
        if khoan_ids:
            gold[qid] = khoan_ids
    return gold


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
    gold = load_gold_khoan(HELDOUT_LABELS_PATH, CONFIDENCE_TRUST)
    print(f"So cau co nhan Khoan dang tin (confidence>={CONFIDENCE_TRUST}): {len(gold)}", flush=True)

    t0 = time.time()
    n_hit_50 = 0
    n_hit_bm25_khoan_alone = 0  # de kiem dieu kien mo Phase 6 (RRF)
    n_scored = 0
    for qid, gold_khoan_ids in gold.items():
        if qid not in qid_to_qemb:
            continue
        question = train[qid]["question"]
        units = search_units_hybrid_plus_bm25_khoan(
            bm25_doc_index, bm25_khoan_index, question, parsed_corpus,
            corpus_emb, corpus_unit_ids, emb_index, qid_to_qemb[qid],
        )
        unit_ids_out = {u["unit_id"] for u in units}

        khoan_scored = retrieve.search(bm25_khoan_index, question, top_k=1)
        bm25_khoan_top1 = {str(khoan_scored[0]["context_id"])} if khoan_scored else set()

        n_scored += 1
        if gold_khoan_ids & unit_ids_out:
            n_hit_50 += 1
        if gold_khoan_ids & bm25_khoan_top1:
            n_hit_bm25_khoan_alone += 1
        if n_scored % 50 == 0:
            print(f"  ... {n_scored}/{len(gold)}, {time.time() - t0:.1f}s", flush=True)

    recall_at_50 = n_hit_50 / n_scored if n_scored else 0.0
    hit1_bm25_khoan_alone = n_hit_bm25_khoan_alone / n_scored if n_scored else 0.0
    print(f"\nSo cau cham duoc: {n_scored}/{len(gold)}", flush=True)
    print(f"\n=== KET QUA ===")
    print(f"Recall@50 cap Khoan (bi-encoder moi, CHUA co BM25-Khoan) : {BASELINE_RECALL_AT_50:.1%}")
    print(f"Recall@50 cap Khoan (+ UNION BM25-Khoan)                : {recall_at_50:.1%}")
    diff = recall_at_50 - BASELINE_RECALL_AT_50
    print(f"Chenh lech: {diff:+.1%}")
    if recall_at_50 < BASELINE_RECALL_AT_50:
        print("=> AM -> BO HAN huong nay.")
    elif diff >= 0.02:
        print("=> Tang >=2 diem% -> GIU, ap dung vao production.")
    else:
        print("=> Trong khoang +0% den +2% -> HOA, can nhac theo chi phi (CPU, gan nhu mien phi -> co the van giu).")

    print(f"\nHit@1 rieng nhanh BM25-Khoan (khong union): {hit1_bm25_khoan_alone:.1%}")
    print("Dieu kien mo Phase 6 (RRF): can Hit@1 nhanh nay >= ~30% (theo PLAN_KE_THUA_TEAM_IR.md).")
    print("  -> " + ("DU DIEU KIEN, co the thu Phase 6." if hit1_bm25_khoan_alone >= 0.30 else "CHUA DU, dung o Phase 3."))


if __name__ == "__main__":
    main()
