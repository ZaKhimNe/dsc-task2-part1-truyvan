"""
Phase 2b — đo recall@50 cấp Khoản với bi-encoder MỚI (AITeamVN/Vietnamese_Embedding)
trên 198 câu heldout (confidence>=0.6), so với ngưỡng đã đặt trước (86,4% hiện tại,
PLAN_NANG_CAP.md Phase 2b): rổ mới < 86,4% -> bỏ ngay; >= 88,4% (+2) -> đi tiếp.

Dùng CHÍNH XÁC cùng logic sinh candidate với `build_layer3_candidates_heldout.py`
(search_units_hybrid, TOP_K_DOCS_LAYER1=100, DENSE_TOP_M=300, TOP_K_OUT=50) — chỉ
đổi nguồn embedding từ `outputs/layer2/` (BAAI/bge-m3) sang `../layer2_embeddings/`
(AITeamVN/Vietnamese_Embedding, vừa tải từ Kaggle) — để so sánh công bằng, không
đổi tham số nào khác ngoài đúng 1 biến đang test.

KHÔNG ghi đè `kaggle_layer3/upload/layer3_candidates_heldout.jsonl` (đang dùng
cho so sánh reranker cũ/mới ở Phase 2a) — ghi ra file candidate riêng.

Chạy: python pipeline/eval_layer2_bienconder_heldout.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config, io_utils
from src.b2_retrieval import retrieve

NEW_LAYER2_DIR = config.REPO_ROOT / "layer2_embeddings"  # AITeamVN/Vietnamese_Embedding, vừa tải
HELDOUT_LABELS_PATH = config.OUTPUTS_DIR / "b0_labels_train_heldout.json"
CONFIDENCE_TRUST = config.B0_CONFIDENCE_TRUST
TOP_K_DOCS_LAYER1 = 100
DENSE_TOP_M = 300
TOP_K_OUT = 50
OUT_CANDIDATES_PATH = config.OUTPUTS_DIR / "layer3_candidates_heldout_aiteamvn.jsonl"
BASELINE_RECALL_AT_50 = 0.864  # đã đo với BAAI/bge-m3, xem PLAN_NANG_CAP.md Phase 2b


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
    gold = load_gold_khoan(HELDOUT_LABELS_PATH, CONFIDENCE_TRUST)
    print(f"So cau co nhan Khoan dang tin (confidence>={CONFIDENCE_TRUST}): {len(gold)}", flush=True)

    n_missing_qemb = sum(1 for qid in gold if qid not in qid_to_qemb)
    print(f"  -> {n_missing_qemb} thieu embedding cau hoi (bo qua)", flush=True)

    OUT_CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_hit_50 = 0
    n_scored = 0
    with open(OUT_CANDIDATES_PATH, "w", encoding="utf-8") as f_out:
        for qid, gold_khoan_ids in gold.items():
            if qid not in qid_to_qemb:
                continue
            question = train[qid]["question"]
            units = retrieve.search_units_hybrid(
                bm25_index, question, parsed_corpus,
                corpus_emb, corpus_unit_ids, emb_index, qid_to_qemb[qid],
                top_k_docs=TOP_K_DOCS_LAYER1, dense_top_m=DENSE_TOP_M, top_k_out=TOP_K_OUT,
            )
            unit_ids_out = [u["unit_id"] for u in units]
            f_out.write(json.dumps({"qid": qid, "unit_ids": unit_ids_out}, ensure_ascii=False) + "\n")

            n_scored += 1
            if gold_khoan_ids & set(unit_ids_out):
                n_hit_50 += 1
            if n_scored % 50 == 0:
                print(f"  ... {n_scored}/{len(gold)}, {time.time() - t0:.1f}s", flush=True)

    recall_at_50 = n_hit_50 / n_scored if n_scored else 0.0
    print(f"\nDa ghi candidate -> {OUT_CANDIDATES_PATH}", flush=True)
    print(f"So cau cham duoc: {n_scored}/{len(gold)}", flush=True)
    print(f"\n=== KET QUA ===")
    print(f"Recall@50 cap Khoan (BAAI/bge-m3, da do truoc)   : {BASELINE_RECALL_AT_50:.1%}")
    print(f"Recall@50 cap Khoan (AITeamVN/Vietnamese_Embedding): {recall_at_50:.1%}")
    diff = recall_at_50 - BASELINE_RECALL_AT_50
    print(f"Chenh lech: {diff:+.1%}")
    if recall_at_50 < BASELINE_RECALL_AT_50:
        print("=> DUOI nguong hien tai -> theo ke hoach da dat truoc: BO NGAY, khong dung ban embedding nay.")
    elif diff >= 0.02:
        print("=> Tang >=2 diem% -> DI TIEP (Phase 2b thanh cong, nen thay production).")
    else:
        print("=> Trong khoang +0% den +2% -> HOA, giu ban cu (khong du de tin la cai thien that).")


if __name__ == "__main__":
    main()
