"""
Đánh giá đóng góp thật của Layer 3 (cross-encoder reranker): so Hit@K CẤP VĂN
BẢN TRƯỚC rerank (thứ tự dense gốc trong `layer3_candidates_train.jsonl`,
đã gửi lên Kaggle) và SAU rerank (`outputs/layer3/L3_rerank_train.jsonl`,
kết quả Kaggle trả về) — trên cùng 7.000 câu `data_retrieve/train.json`
(nhãn context_id sạch, cho sẵn, độc lập — không qua B0/B2).

Đo cấp văn bản (không phải cấp Khoản) vì đây là nhãn SẠCH duy nhất có sẵn
cho toàn bộ 7.000 câu — nhãn B0 cấp Khoản chỉ phủ được phần nhỏ đã gán, và
cấp Khoản của tập câu hỏi này (data_retrieve) không tồn tại (chỉ có
context_id, không có quote để B0 gán tiếp).

Với mỗi câu, quy về context_id (collapse từ unit_id) rồi tính Hit@K, giữ thứ
tự XUẤT HIỆN ĐẦU TIÊN của mỗi context_id trong danh sách gốc (không sắp xếp
lại theo gì khác ngoài thứ tự đã cho).

Chạy: python pipeline/eval_layer3_rerank.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config

CANDIDATES_PATH = Path("kaggle_layer3/upload/layer3_candidates_train.jsonl")
RERANK_PATH = Path("outputs/layer3/L3_rerank_train.jsonl")
GOLD_PATH = config.DATA_RETRIEVE_DIR / "train.json"
KS = [1, 3, 5, 10, 20, 30, 50]


def collapse_to_docs(unit_ids: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for uid in unit_ids:
        cid = uid.split("_")[0]
        if cid not in seen:
            seen[cid] = None
    return list(seen.keys())


def load_before() -> dict[str, list[str]]:
    """Thứ tự TRƯỚC rerank = thứ tự candidate đã gửi Kaggle (dense-ranked gốc)."""
    result = {}
    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            result[r["qid"]] = [c["context_id"] for c in r["candidates"]]
    return result


def load_after() -> dict[str, list[str]]:
    result = {}
    with open(RERANK_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            unit_ids = [u["unit_id"] for u in r["ranked_units"]]
            result[r["qid"]] = collapse_to_docs(unit_ids)
    return result


def hit_at_k(runs: dict[str, list[str]], gold: dict[str, list[str]], ks: list[int]) -> dict[int, float]:
    out = {}
    n = len(runs)
    for k in ks:
        n_hit = sum(
            1 for qid, docs in runs.items()
            if set(docs[:k]) & set(str(c) for c in gold.get(qid, {}).get("answer", []))
        )
        out[k] = n_hit / n
    return out


def mrr(runs: dict[str, list[str]], gold: dict[str, list[str]]) -> float:
    total = 0.0
    n = len(runs)
    for qid, docs in runs.items():
        gold_set = set(str(c) for c in gold.get(qid, {}).get("answer", []))
        rank = next((i + 1 for i, d in enumerate(docs) if d in gold_set), None)
        if rank:
            total += 1.0 / rank
    return total / n


def main():
    with open(GOLD_PATH, encoding="utf-8") as f:
        gold = json.load(f)

    before = load_before()
    after = load_after()
    print(f"n câu (before)={len(before)}, n câu (after)={len(after)}, n gold={len(gold)}")

    hit_before = hit_at_k(before, gold, KS)
    hit_after = hit_at_k(after, gold, KS)
    mrr_before = mrr(before, gold)
    mrr_after = mrr(after, gold)

    print(f"\n{'K':>4} | {'Trước rerank (dense)':>22} | {'Sau rerank (Layer 3)':>22} | {'Chênh lệch':>12}")
    print("-" * 70)
    for k in KS:
        diff = hit_after[k] - hit_before[k]
        print(f"{k:>4} | {hit_before[k]:>21.1%} | {hit_after[k]:>21.1%} | {diff:>+11.1%}")
    print("-" * 70)
    print(f"{'MRR':>4} | {mrr_before:>21.4f} | {mrr_after:>21.4f} | {mrr_after - mrr_before:>+11.4f}")


if __name__ == "__main__":
    main()
