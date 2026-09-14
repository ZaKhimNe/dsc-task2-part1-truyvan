"""
Error analysis — 800 câu heldout, Layer 3 rerank: phân loại NGUYÊN NHÂN các
case Layer 3 vẫn miss (gold Khoản không nằm trong top-K sau rerank), để biết
nên đầu tư sửa B1/B2 (retrieval) hay Layer 3 (reranker) tiếp theo.

Gold: UNION 2 nguồn nhãn ĐỘC LẬP (xem outputs/LAYER3_EVALUATION_SUMMARY.md) —
B0 (auto-label, confidence>=0.6) + citation-match (parse từ citation_metadata,
data123) — chỉ unit_type="khoan" cả 2 nguồn.

Phân loại (chỉ xét case Hit@HIT_K sau Layer 3 = FALSE):
  Loại A (retrieval miss) — gold KHÔNG nằm trong top-50 TRƯỚC Layer 3 (candidate
    gốc từ search_units_hybrid) -> lỗi B1/B2, Layer 3 không thể cứu vì gold
    chưa từng vào candidate pool.
  Loại B (reranker miss)  — gold CÓ trong top-50 trước nhưng bị Layer 3 xếp
    hạng xuống ngoài top-HIT_K sau -> lỗi cross-encoder rerank.

Chạy: python pipeline/error_analysis_heldout.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config, io_utils
from src.b2_retrieval.retrieve import get_unit_text

CANDIDATES_PATH = Path("kaggle_layer3/upload/layer3_candidates_heldout.jsonl")
RERANK_PATH = Path("outputs/layer3/L3_rerank_heldout.jsonl")
B0_LABELS_PATH = config.OUTPUTS_DIR / "b0_labels_train_heldout.json"
CITATION_LABELS_DIR = config.OUTPUTS_DIR
OUT_PATH = config.OUTPUTS_DIR / "error_analysis_heldout.json"
HIT_K = 10
TOP_SHOW = 5


def load_heldout_ids() -> set[str]:
    with open(B0_LABELS_PATH, encoding="utf-8") as f:
        return set(json.load(f)["heldout_ids"])


def load_b0_gold() -> dict[str, set[str]]:
    with open(B0_LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)["labels"]
    gold = {}
    for qid, spans in labels.items():
        khoan_ids = {
            s["khoan_id"] for s in spans
            if s["unit_type"] == "khoan" and s["confidence"] >= config.B0_CONFIDENCE_TRUST
        }
        if khoan_ids:
            gold[qid] = khoan_ids
    return gold


def load_citation_gold(heldout_ids: set[str]) -> dict[str, set[str]]:
    gold = {}
    for split in ["train", "validation", "test"]:
        path = CITATION_LABELS_DIR / f"citation_labels_{split}.json"
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        for r in records:
            qid = r["qid"]
            if qid not in heldout_ids:
                continue
            if r["unit_type"] == "khoan":
                gold.setdefault(qid, set()).add(r["unit_id"])
    return gold


def merge_gold(b0_gold: dict, citation_gold: dict) -> dict[str, set[str]]:
    """Union 2 nguồn — câu có cả 2 thì gộp set khoan_id lại."""
    merged: dict[str, set[str]] = {}
    for qid, units in b0_gold.items():
        merged.setdefault(qid, set()).update(units)
    for qid, units in citation_gold.items():
        merged.setdefault(qid, set()).update(units)
    return merged


def load_before() -> dict[str, dict]:
    """qid -> {"question", "candidates": [{"unit_id",...}, ...]} (record gốc)."""
    result = {}
    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            result[r["qid"]] = r
    return result


def load_after() -> dict[str, list[dict]]:
    """qid -> list {"unit_id","context_id","score"} (top-50 sau rerank, đã sort)."""
    result = {}
    with open(RERANK_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            result[r["qid"]] = r["ranked_units"]
    return result


def main():
    print("Đang load gold labels + candidates...", flush=True)
    heldout_ids = load_heldout_ids()
    b0_gold = load_b0_gold()
    citation_gold = load_citation_gold(heldout_ids)
    gold = merge_gold(b0_gold, citation_gold)
    before = load_before()
    after = load_after()
    parsed_corpus = io_utils.load_parsed_corpus()

    n_overlap = len(set(b0_gold) & set(citation_gold))
    print(f"Union gold: {len(gold)} câu (B0={len(b0_gold)}, citation={len(citation_gold)}, "
          f"overlap={n_overlap})", flush=True)

    cases = []
    n_hit = 0
    n_loai_a = 0
    n_loai_b = 0

    for qid, gold_units in gold.items():
        before_record = before.get(qid, {})
        before_units = before_record.get("candidates", [])
        question = before_record.get("question", "")
        after_units = after.get(qid, [])

        after_top_ids = [u["unit_id"] for u in after_units[:HIT_K]]
        if gold_units & set(after_top_ids):
            n_hit += 1
            continue

        before_ids = [c["unit_id"] for c in before_units]
        in_before = gold_units & set(before_ids)

        if not in_before:
            loai = "A"
            n_loai_a += 1
            rank_before = None
        else:
            loai = "B"
            n_loai_b += 1
            matched_gold_id = next(iter(in_before))
            rank_before = before_ids.index(matched_gold_id) + 1

        after_all_ids = [u["unit_id"] for u in after_units]
        rank_after = None
        for gu in gold_units:
            if gu in after_all_ids:
                rank_after = after_all_ids.index(gu) + 1
                break

        top5 = [
            {
                "unit_id": u["unit_id"],
                "score": u["score"],
                "text": get_unit_text(parsed_corpus, u["unit_id"])[:150],
            }
            for u in after_units[:TOP_SHOW]
        ]

        cases.append({
            "qid": qid,
            "question": question,
            "loai": loai,
            "rank_before": rank_before,
            "rank_after": rank_after,
            "gold_units": {gu: get_unit_text(parsed_corpus, gu)[:200] for gu in gold_units},
            "top5_after": top5,
        })

    print(f"\n=== KẾT QUẢ (Hit@{HIT_K}) ===", flush=True)
    print(f"Tổng câu có gold: {len(gold)}", flush=True)
    print(f"Hit: {n_hit} ({n_hit / len(gold):.1%})", flush=True)
    print(f"Thất bại: {len(cases)} ({len(cases) / len(gold):.1%})", flush=True)
    if cases:
        print(f"  Loại A (retrieval miss — gold KHÔNG có trong top-50 trước): "
              f"{n_loai_a} ({n_loai_a / len(cases):.1%} trong số case thất bại)", flush=True)
        print(f"  Loại B (reranker miss — gold CÓ trong top-50 trước, bị đẩy ngoài top-{HIT_K} sau): "
              f"{n_loai_b} ({n_loai_b / len(cases):.1%} trong số case thất bại)", flush=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "n_total_gold": len(gold), "n_hit": n_hit, "n_fail": len(cases),
                "n_loai_a": n_loai_a, "n_loai_b": n_loai_b,
            },
            "cases": cases,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
