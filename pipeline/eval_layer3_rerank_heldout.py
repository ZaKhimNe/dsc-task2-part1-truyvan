"""
Đánh giá Layer 3 ở ĐÚNG CẤP KHOẢN, trên TẬP GIỮ KÍN (800 câu `data/train.json`,
chưa từng dùng để tune bất cứ gì) — bước xác nhận cuối cùng trước khi chốt
Layer 3 vào kiến trúc chính thức. Cùng cách làm với `eval_layer3_rerank.py`
(so Hit@K TRƯỚC/SAU rerank), nhưng đổi gold từ context_id cấp văn bản
(`data_retrieve`) sang nhãn B0 CẤP KHOẢN (`outputs/b0_labels_train_heldout.json`,
lọc confidence>=0.6 — n=198 câu đáng tin).

Chạy: python pipeline/eval_layer3_rerank_heldout.py
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config

CANDIDATES_PATH = Path("kaggle_layer3/upload/layer3_candidates_heldout.jsonl")
RERANK_PATH = Path("outputs/layer3/L3_rerank_heldout.jsonl")
LABELS_PATH = config.OUTPUTS_DIR / "b0_labels_train_heldout.json"
KS = [1, 3, 5, 10, 20, 30, 50]
CONFIDENCE_TRUST = config.B0_CONFIDENCE_TRUST
N_BOOTSTRAP = 2000
SEED = 42


def load_gold() -> dict[str, set[str]]:
    """gold[qid] = tập khoan_id đáng tin cậy (confidence>=0.6, unit_type='khoan')."""
    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)["labels"]
    gold = {}
    for qid, spans in labels.items():
        khoan_ids = {
            s["khoan_id"] for s in spans
            if s["unit_type"] == "khoan" and s["confidence"] >= CONFIDENCE_TRUST
        }
        if khoan_ids:
            gold[qid] = khoan_ids
    return gold


def load_before() -> dict[str, list[str]]:
    """Thứ tự TRƯỚC rerank = thứ tự candidate đã gửi Kaggle (dense-ranked gốc)."""
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
    gold = load_gold()
    before = load_before()
    after = load_after()
    print(f"n câu giữ kín có nhãn Khoản đáng tin (confidence>={CONFIDENCE_TRUST}): {len(gold)}")

    # sàn nhiễu: bootstrap CI của Hit@3 riêng "after" để biết biên độ dao động tự nhiên ở n này
    print(f"\n{'K':>4} | {'Trước (dense)':>15} | {'Sau (Layer 3)':>15} | {'Chênh lệch':>12} | {'95% CI chênh lệch':>20}")
    print("-" * 85)
    for k in KS:
        h_before = hit_at_k(before, gold, k)
        h_after = hit_at_k(after, gold, k)
        diff = h_after - h_before
        lo, hi = bootstrap_ci_diff(before, after, gold, k, N_BOOTSTRAP, SEED)
        sig = "***" if (lo > 0 or hi < 0) else ""
        print(f"{k:>4} | {h_before:>14.1%} | {h_after:>14.1%} | {diff:>+11.1%} | [{lo:+.1%}, {hi:+.1%}] {sig}")
    print("-" * 85)
    mrr_before = mrr(before, gold)
    mrr_after = mrr(after, gold)
    print(f"{'MRR':>4} | {mrr_before:>14.4f} | {mrr_after:>14.4f} | {mrr_after - mrr_before:>+11.4f}")
    print("\n(*** = khoảng tin cậy 95% không chứa 0, chênh lệch có ý nghĩa thống kê)")


if __name__ == "__main__":
    main()
