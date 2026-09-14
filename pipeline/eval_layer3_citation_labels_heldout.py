"""
Đánh giá Layer 3 ở cấp KHOẢN trên 800 câu heldout, dùng nhãn từ citation_metadata
(Team C) thay vì nhãn B0 — để so sánh chất lượng 2 nguồn nhãn và đo Layer 3 với
nhãn chính xác hơn (parse từ câu trích dẫn, không suy đoán mờ).

So sánh 3 phiên bản:
  1. B0 (nhãn B0, confidence>=0.6, n~198)
  2. Citation-match (nhãn Team C, parse từ citation_metadata, n~800)
  3. Đối chiếu: B0 vs citation-match trên phần chồng lấp

Chạy: python pipeline/eval_layer3_citation_labels_heldout.py
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config

CANDIDATES_PATH = Path("kaggle_layer3/upload/layer3_candidates_heldout.jsonl")
RERANK_PATH = Path("outputs/layer3/L3_rerank_heldout.jsonl")
B0_LABELS_PATH = config.OUTPUTS_DIR / "b0_labels_train_heldout.json"
CITATION_LABELS_DIR = config.OUTPUTS_DIR
KS = [1, 3, 5, 10, 20, 30, 50]
N_BOOTSTRAP = 2000
SEED = 42


def load_heldout_ids() -> set[str]:
    """Load 800 qid từ tập giữ kín."""
    with open(B0_LABELS_PATH, encoding="utf-8") as f:
        return set(json.load(f)["heldout_ids"])


def load_b0_gold() -> dict[str, set[str]]:
    """gold[qid] = tập khoan_id từ B0 (confidence>=0.6, unit_type='khoan')."""
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
    """gold[qid] = tập unit_id từ citation_labels (unit_type='khoan'), chỉ 800 heldout."""
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
                if qid not in gold:
                    gold[qid] = set()
                gold[qid].add(r["unit_id"])
    return gold


def load_before() -> dict[str, list[str]]:
    result = {}
    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            result[r["qid"]] = [c["unit_id"] for c in r["candidates"]]
    return result


def load_after() -> dict[str, list[str]]:
    result = {}
    with open(RERANK_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            result[r["qid"]] = [u["unit_id"] for u in r["ranked_units"]]
    return result


def hit_at_k(runs: dict[str, list[str]], gold: dict[str, set[str]], k: int) -> float:
    n_hit = sum(1 for qid, g in gold.items() if g & set(runs[qid][:k]))
    return n_hit / len(gold)


def bootstrap_ci_diff(runs_a, runs_b, gold: dict[str, set[str]], k: int, n_boot: int, seed: int):
    qids = list(gold.keys())
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        sample = [qids[rng.randrange(len(qids))] for _ in range(len(qids))]
        hit_a = sum(1 for qid in sample if gold[qid] & set(runs_a[qid][:k])) / len(sample)
        hit_b = sum(1 for qid in sample if gold[qid] & set(runs_b[qid][:k])) / len(sample)
        diffs.append(hit_b - hit_a)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot)]
    return lo, hi


def mrr(runs: dict[str, list[str]], gold: dict[str, set[str]]) -> float:
    total = 0.0
    for qid, g in gold.items():
        rank = next((i + 1 for i, u in enumerate(runs[qid]) if u in g), None)
        if rank:
            total += 1.0 / rank
    return total / len(gold)


def main():
    heldout_ids = load_heldout_ids()
    b0_gold = load_b0_gold()
    citation_gold = load_citation_gold(heldout_ids)
    before = load_before()
    after = load_after()

    print("=" * 100, flush=True)
    print("ĐÁNH GIÁ LAYER 3 TRÊN 800 CÂU HELDOUT — B0 vs CITATION-MATCH", flush=True)
    print("=" * 100, flush=True)
    print(f"\nTập giữ kín 800 câu từ `data/train.json`", flush=True)
    print(f"  B0 nhãn (confidence>=0.6, unit_type='khoan'): {len(b0_gold)} câu", flush=True)
    print(f"  Citation-match (từ data123/, unit_type='khoan'): {len(citation_gold)} câu", flush=True)
    print(f"  Chồng lấp (cả 2 có): {len(set(b0_gold.keys()) & set(citation_gold.keys()))} câu", flush=True)

    # Đánh giá trên B0 gold
    print("\n" + "=" * 100, flush=True)
    print("KÊNH 1: ĐÁNH GIÁ TRÊN B0 GOLD (n={})".format(len(b0_gold)), flush=True)
    print("=" * 100, flush=True)
    print(f"\n{'K':>4} | {'Trước (dense)':>15} | {'Sau (Layer 3)':>15} | {'Chênh lệch':>12} | {'95% CI chênh lệch':>20}")
    print("-" * 85)
    for k in KS:
        h_before = hit_at_k(before, b0_gold, k)
        h_after = hit_at_k(after, b0_gold, k)
        diff = h_after - h_before
        lo, hi = bootstrap_ci_diff(before, after, b0_gold, k, N_BOOTSTRAP, SEED)
        sig = "***" if (lo > 0 or hi < 0) else ""
        print(f"{k:>4} | {h_before:>14.1%} | {h_after:>14.1%} | {diff:>+11.1%} | [{lo:+.1%}, {hi:+.1%}] {sig}")
    print("-" * 85)
    mrr_before_b0 = mrr(before, b0_gold)
    mrr_after_b0 = mrr(after, b0_gold)
    print(f"{'MRR':>4} | {mrr_before_b0:>14.4f} | {mrr_after_b0:>14.4f} | {mrr_after_b0 - mrr_before_b0:>+11.4f}")

    # Đánh giá trên citation gold
    print("\n" + "=" * 100, flush=True)
    print("KÊNH 2: ĐÁNH GIÁ TRÊN CITATION-MATCH GOLD (n={})".format(len(citation_gold)), flush=True)
    print("=" * 100, flush=True)
    print(f"\n{'K':>4} | {'Trước (dense)':>15} | {'Sau (Layer 3)':>15} | {'Chênh lệch':>12} | {'95% CI chênh lệch':>20}")
    print("-" * 85)
    for k in KS:
        h_before = hit_at_k(before, citation_gold, k)
        h_after = hit_at_k(after, citation_gold, k)
        diff = h_after - h_before
        lo, hi = bootstrap_ci_diff(before, after, citation_gold, k, N_BOOTSTRAP, SEED)
        sig = "***" if (lo > 0 or hi < 0) else ""
        print(f"{k:>4} | {h_before:>14.1%} | {h_after:>14.1%} | {diff:>+11.1%} | [{lo:+.1%}, {hi:+.1%}] {sig}")
    print("-" * 85)
    mrr_before_cite = mrr(before, citation_gold)
    mrr_after_cite = mrr(after, citation_gold)
    print(f"{'MRR':>4} | {mrr_before_cite:>14.4f} | {mrr_after_cite:>14.4f} | {mrr_after_cite - mrr_before_cite:>+11.4f}")

    # So sánh B0 vs citation trên phần chồng lấp
    overlap_qids = set(b0_gold.keys()) & set(citation_gold.keys())
    if overlap_qids:
        overlap_b0_gold = {qid: b0_gold[qid] for qid in overlap_qids}
        overlap_citation_gold = {qid: citation_gold[qid] for qid in overlap_qids}
        print("\n" + "=" * 100, flush=True)
        print("KÊNH 3: ĐỐI CHIẾU B0 vs CITATION (phần chồng lấp, n={})".format(len(overlap_qids)), flush=True)
        print("=" * 100, flush=True)
        print(f"\n{'K':>4} | {'B0 gold (sau L3)':>15} | {'Citation gold (sau L3)':>22} | {'Chênh lệch':>12}")
        print("-" * 65)
        for k in KS:
            h_b0 = hit_at_k(after, overlap_b0_gold, k)
            h_cite = hit_at_k(after, overlap_citation_gold, k)
            diff = h_cite - h_b0
            print(f"{k:>4} | {h_b0:>14.1%} | {h_cite:>21.1%} | {diff:>+11.1%}")
        print("-" * 65)

    print("\n(*** = khoảng tin cậy 95% không chứa 0, chênh lệch có ý nghĩa thống kê)", flush=True)


if __name__ == "__main__":
    main()
