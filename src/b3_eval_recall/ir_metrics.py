"""
B3 — Metric IR tổng quát (Hit@K, Recall@K micro/strict, MRR, nDCG, rank percentile).

Dùng để đo B2 bằng nhãn "sạch" từ data_retrieve (context_id đúng cho sẵn,
KHÔNG qua suy luận B0/Smith-Waterman) — xem docs/EXPERIMENT_LOG.md entry
[B2] 12/08 (chiến lược ground truth 4 tầng). Khác `eval_recall.py` (dùng
nhãn B0 `MatchedSpan`, gắn với schema Khoản/unit_type riêng của B0) — module
này TỔNG QUÁT, chỉ cần (qid -> ranked id list) + (qid -> gold id list),
không phụ thuộc schema nào của B0/B1/B2.

run: dict[qid, list[id]] — ĐÃ sắp xếp giảm dần theo score (chỉ cần thứ tự).
gold: dict[qid, list[id]] — có thể nhiều gold id/câu (đa nguồn, vd 7.9% câu
      trong data_retrieve có >1 context_id đúng).

Câu hỏi không có gold (list rỗng) bị loại khỏi mẫu số MỌI metric (không có
gì để so) — n_no_gold báo riêng, không trộn vào recall.

Vì sao nhiều metric, không chỉ Recall@K:
  - Hit@K: "có tìm được ít nhất 1 đúng không" — cách gọi phổ biến nhưng hay
    bị nhầm là Recall@K.
  - Recall@K micro: đúng nghĩa IR — quan trọng khi 1 câu có NHIỀU gold
    (đa nguồn), không phải chỉ cần 1 đúng là đủ.
  - Recall@K strict: TẤT CẢ gold phải có mặt trong top-K — QA cần khi answer
    trích ≥2 văn bản (thiếu 1 nguồn vẫn là trả lời thiếu).
  - MRR: phân biệt "đúng ở hạng 15" (Layer 2/3 rerank có thể cứu) với "đúng
    không có trong top-K nào cả" (phải thêm dense retrieval, rerank vô ích)
    — đây là con số quyết định cho B5.
  - nDCG@K: xếp hạng có tính vị trí, hợp cho câu đa gold.
  - rank_percentiles: p50/p90/p95/p99 của rank gold đầu tiên (trong số câu
    TÌM ĐƯỢC) — dùng để chọn K bơm vào Layer 2 bằng dữ liệu thật, không đoán.
"""
from __future__ import annotations

import math


def _gold_ranks(ranked_ids: list, gold_ids: set) -> list[int]:
    """Rank (1-indexed) của các gold_id xuất hiện trong ranked_ids, sắp tăng
    dần. Gold id không xuất hiện trong ranked_ids -> bỏ qua (không tính vào
    K nào, kể cả K lớn nhất đã lấy — nghĩa là "không tìm thấy trong phạm vi
    đã retrieve", có thể do bị cắt ở max(ks) hoặc bị index bỏ sót thật)."""
    return sorted(rank for rank, rid in enumerate(ranked_ids, start=1) if rid in gold_ids)


def compute_query_stats(run: dict, gold: dict) -> dict:
    """Tính sẵn (gold_ranks, n_gold) cho mỗi câu hỏi 1 lần — mọi metric/K
    dùng chung, không quét lại run nhiều lần."""
    per_query: dict[str, dict] = {}
    n_no_gold = 0
    for qid, gold_ids in gold.items():
        if not gold_ids:
            n_no_gold += 1
            continue
        ranked = run.get(qid, [])
        per_query[qid] = {"gold_ranks": _gold_ranks(ranked, set(gold_ids)), "n_gold": len(gold_ids)}
    return {"per_query": per_query, "n_no_gold": n_no_gold}


def hit_at_k(query_stats: dict, k: int) -> float:
    qs = query_stats["per_query"]
    if not qs:
        return 0.0
    return sum(1 for s in qs.values() if s["gold_ranks"] and s["gold_ranks"][0] <= k) / len(qs)


def recall_at_k_micro(query_stats: dict, k: int) -> float:
    qs = query_stats["per_query"]
    total_gold = sum(s["n_gold"] for s in qs.values())
    if total_gold == 0:
        return 0.0
    found = sum(sum(1 for r in s["gold_ranks"] if r <= k) for s in qs.values())
    return found / total_gold


def recall_at_k_strict(query_stats: dict, k: int) -> float:
    qs = query_stats["per_query"]
    if not qs:
        return 0.0
    hits = sum(
        1
        for s in qs.values()
        if len(s["gold_ranks"]) == s["n_gold"] and (not s["gold_ranks"] or s["gold_ranks"][-1] <= k)
    )
    return hits / len(qs)


def mrr(query_stats: dict, k: int | None = None) -> float:
    qs = query_stats["per_query"]
    if not qs:
        return 0.0
    total = 0.0
    for s in qs.values():
        ranks = s["gold_ranks"] if k is None else [r for r in s["gold_ranks"] if r <= k]
        if ranks:
            total += 1.0 / ranks[0]
    return total / len(qs)


def ndcg_at_k(query_stats: dict, k: int) -> float:
    """nDCG@K, binary relevance (gain=1 cho gold, 0 cho non-gold)."""
    qs = query_stats["per_query"]
    if not qs:
        return 0.0
    total = 0.0
    for s in qs.values():
        dcg = sum(1.0 / math.log2(r + 1) for r in s["gold_ranks"] if r <= k)
        ideal_n = min(s["n_gold"], k)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
        if idcg > 0:
            total += dcg / idcg
    return total / len(qs)


def rank_percentiles(query_stats: dict, percentiles: tuple[int, ...] = (50, 90, 95, 99)) -> dict:
    """Percentile của rank gold ĐẦU TIÊN, chỉ trong số câu tìm được (tách
    biệt khỏi coverage — "nếu tìm được thì thường ở hạng bao nhiêu")."""
    qs = query_stats["per_query"]
    firsts = sorted(s["gold_ranks"][0] for s in qs.values() if s["gold_ranks"])
    result: dict = {"n_found_somewhere": len(firsts), "n_total": len(qs)}
    if not firsts:
        for p in percentiles:
            result[p] = None
        return result
    for p in percentiles:
        idx = min(len(firsts) - 1, int(len(firsts) * p / 100))
        result[p] = firsts[idx]
    return result


def evaluate(run: dict, gold: dict, ks: list[int]) -> dict:
    """Hàm chính: tính toàn bộ bộ metric cho nhiều K cùng lúc từ 1 run đã có
    (không tính lại retrieval — run nên đã lấy sẵn top-max(ks))."""
    query_stats = compute_query_stats(run, gold)
    per_k = [
        {
            "k": k,
            "hit_at_k": hit_at_k(query_stats, k),
            "recall_micro": recall_at_k_micro(query_stats, k),
            "recall_strict": recall_at_k_strict(query_stats, k),
            "mrr_at_k": mrr(query_stats, k),
            "ndcg_at_k": ndcg_at_k(query_stats, k),
        }
        for k in ks
    ]
    return {
        "n_queries_evaluated": len(query_stats["per_query"]),
        "n_no_gold": query_stats["n_no_gold"],
        "mrr_full": mrr(query_stats, k=None),
        "rank_percentiles": rank_percentiles(query_stats),
        "per_k": per_k,
    }
