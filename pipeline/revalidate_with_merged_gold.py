"""
Kiem chung lai 2 quyet dinh da chot (doi reranker Phase 2a, doi bi-encoder
Phase 2b) bang bo nhan gop MOI (gold_labels_merged_heldout.json, n=286) thay
vi chi dung nhan B0 rieng (n=198) — xem ket luan cu co con dung voi co mau
lon hon, da kiem chung cheo hay khong.

Chay: python pipeline/revalidate_with_merged_gold.py
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config

MERGED_GOLD_PATH = config.OUTPUTS_DIR / "gold_labels_merged_heldout.json"


def load_gold() -> dict[str, set[str]]:
    data = json.loads(MERGED_GOLD_PATH.read_text(encoding="utf-8"))
    gold = {}
    for qid, info in data.items():
        if info["source"] == "disagree":
            continue
        gold[qid] = set(info["khoan_ids"])
    return gold


def load_ranked_units(path: Path) -> dict[str, list[str]]:
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            result[r["qid"]] = [u["unit_id"] for u in r["ranked_units"]]
    return result


def load_candidates_as_ranked(path: Path) -> dict[str, list[str]]:
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "unit_ids" in r:
                result[r["qid"]] = r["unit_ids"]
            else:
                result[r["qid"]] = [c["unit_id"] for c in r["candidates"]]
    return result


def hit_at_k(runs, gold, k):
    n_hit = sum(1 for qid, g in gold.items() if qid in runs and g & set(runs[qid][:k]))
    n_scored = sum(1 for qid in gold if qid in runs)
    return n_hit / n_scored if n_scored else 0.0, n_scored


def bootstrap_ci_diff(runs_a, runs_b, gold, k, n_boot=2000, seed=42):
    qids = [q for q in gold if q in runs_a and q in runs_b]
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        sample = [qids[rng.randrange(len(qids))] for _ in range(len(qids))]
        a = sum(1 for qid in sample if gold[qid] & set(runs_a[qid][:k])) / len(sample)
        b = sum(1 for qid in sample if gold[qid] & set(runs_b[qid][:k])) / len(sample)
        diffs.append(b - a)
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]


def main():
    gold = load_gold()
    print(f"So cau trong bo nhan gop (da bo disagree): {len(gold)}\n")

    print("=" * 78)
    print("KIEM CHUNG LAI PHASE 2a — doi reranker (BAAI -> AITeamVN)")
    print("Ca 2 deu dung candidate tu bi-encoder CU (dung dung cau hinh da so sanh truoc)")
    print("=" * 78)
    before = load_ranked_units(config.OUTPUTS_DIR / "layer3" / "L3_rerank_heldout_bge_v2m3.jsonl")
    after = load_ranked_units(config.OUTPUTS_DIR / "layer3" / "L3_rerank_heldout.jsonl")
    for k in [1, 3, 5]:
        h_before, n = hit_at_k(before, gold, k)
        h_after, _ = hit_at_k(after, gold, k)
        lo, hi = bootstrap_ci_diff(before, after, gold, k)
        sig = "***" if (lo > 0 or hi < 0) else ""
        print(f"Hit@{k}: cu={h_before:.1%}  moi={h_after:.1%}  chenh={h_after-h_before:+.1%}  CI=[{lo:+.1%},{hi:+.1%}] {sig}  (n={n})")

    print()
    print("=" * 78)
    print("KIEM CHUNG LAI PHASE 2b — doi bi-encoder (BAAI -> AITeamVN), Recall@50")
    print("=" * 78)
    old_candidates = load_candidates_as_ranked(Path("kaggle_layer3/upload/layer3_candidates_heldout.jsonl"))
    new_candidates = load_candidates_as_ranked(config.OUTPUTS_DIR / "layer3_candidates_heldout_aiteamvn_full.jsonl")
    h_old, n_old = hit_at_k(old_candidates, gold, 50)
    h_new, n_new = hit_at_k(new_candidates, gold, 50)
    lo, hi = bootstrap_ci_diff(old_candidates, new_candidates, gold, 50)
    sig = "***" if (lo > 0 or hi < 0) else ""
    print(f"Recall@50: cu={h_old:.1%}  moi={h_new:.1%}  chenh={h_new-h_old:+.1%}  CI=[{lo:+.1%},{hi:+.1%}] {sig}  (n={n_old})")

    print("\n(*** = khoang tin cay 95% khong chua 0, chenh lech co y nghia thong ke)")
    print("\nSo sanh voi ket qua cu (chi dung B0, n=198):")
    print("  Phase 2a Hit@3: 59.6% -> 68.7% (+9.1%)")
    print("  Phase 2b Recall@50: 86.4% -> 90.4% (+4.0%)")


if __name__ == "__main__":
    main()
