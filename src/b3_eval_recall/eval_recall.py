"""
B3 — Đo Recall & chốt granularity.

Bài toán: đo Recall@K của B2 (retrieval) bằng nhãn thật từ B0 (auto-label).
CHỈ SỐ SỞ HỮU #1 của B (theo tài liệu phân công gốc mục 3.3, "chỉ số chặn
trên tuyệt đối" — document sai thì B4/B6/B7 và cả Generator của C phía sau
đều vô nghĩa).

Có 2 mức đo, dùng cho 2 mục đích khác nhau:
  - `recall_at_k` (cấp VĂN BẢN, context_id): đo B2 Layer 1 (BM25 doc-level)
    — câu hỏi có đủ trong top-K văn bản không, trước khi Expand sang Khoản.
  - `recall_at_k_unit` (cấp KHOẢN/Dieu, unit_id): đo sau Expand/Layer 2/3 —
    dùng để so sánh BM25-only vs +bi-encoder vs +reranker (bằng chứng cho
    B5, xem docs/EXPERIMENT_LOG.md entry [B2] 12/08). "unit đúng" của 1
    MatchedSpan = `khoan_id` nếu unit_type="khoan", ngược lại (dieu_fallback/
    doc_fallback, không có Khoản thật) fallback về chính unit_id mà B0 đã
    gán (dieu_id hoặc context_id) — xem `_true_unit_id`.

Input:
  - labels: {question_id: [MatchedSpan, ...]} — output B0 (label_answer/label_dataset)
  - retrieved: CÙNG bộ question_id với labels, đã lấy sẵn top-K_max (K muốn đo nhỏ hơn K_max)

Output: dict {"recall_at_k", "k", "n_evaluated", "n_no_label", "n_hits"}.

Câu hỏi bị B0 không gán được nhãn nào (list rỗng — answer không tìm được
nguồn tin cậy) BỊ LOẠI khỏi mẫu số Recall@K, không tính là "trượt" (không có
ground truth để so). Tỉ lệ này (n_no_label / tổng) là coverage của B0 — báo
cáo RIÊNG, không gộp lẫn vào Recall@K kẻo đánh giá sai B2.

REBUILD 12/08/2026: `context_id` trong MatchedSpan đổi từ int -> str (khớp
schema B1/B2 mới, context_id vốn dùng làm tiền tố khoan_id/dieu_id dạng
string). `recall_at_k` ép cả 2 phía về str khi so sánh — ScoredContext (B2
Layer 1) vẫn giữ context_id: int (từ io_utils.ContextDoc), không đổi, để
tránh phải sửa dây chuyền ngược lên B2/io_utils chỉ vì việc so sánh ở đây.

TRẠNG THÁI: bản sơ bộ 06/08 đã chạy trên corpus subset có distractor (không
phải full 8,532 văn bản). Số liệu CHÍNH THỨC cần chạy trên full corpus +
full train/warmup — chờ B0 (`build_b0_labels.py`) chạy xong.
"""
from __future__ import annotations

from src.b0_autolabel.label import MatchedSpan


def recall_at_k(
    labels: dict[str, list[MatchedSpan]],
    retrieved: dict[str, list[dict]],
    k: int,
) -> dict:
    """Recall@K cấp VĂN BẢN (B2 Layer 1, trước Expand).

    Recall@K = tỉ lệ câu hỏi (trong số câu hỏi CÓ nhãn) mà ít nhất 1
    context_id đúng xuất hiện trong top-K kết quả `retrieved` (list[dict]
    có key "context_id", vd ScoredContext). So sánh ép cả 2 phía về str
    (xem cảnh báo docstring module về lệch kiểu int/str)."""
    n_no_label = 0
    hits = 0
    n_evaluated = 0
    for qid, spans in labels.items():
        true_ids = {str(m["context_id"]) for m in spans}
        if not true_ids:
            n_no_label += 1
            continue
        retrieved_ids = {str(r["context_id"]) for r in retrieved.get(qid, [])[:k]}
        n_evaluated += 1
        if true_ids & retrieved_ids:
            hits += 1
    recall = hits / n_evaluated if n_evaluated else 0.0
    return {
        "recall_at_k": recall,
        "k": k,
        "n_evaluated": n_evaluated,
        "n_no_label": n_no_label,
        "n_hits": hits,
    }


def recall_at_multiple_k(
    labels: dict[str, list[MatchedSpan]],
    retrieved: dict[str, list[dict]],
    ks: list[int],
) -> list[dict]:
    """Recall@K cấp văn bản cho nhiều giá trị K — tiện cho bảng so sánh.
    `retrieved` nên đã lấy sẵn top-K_max = max(ks)."""
    return [recall_at_k(labels, retrieved, k) for k in ks]


def _true_unit_id(span: MatchedSpan) -> str:
    """ID unit "đúng" của 1 MatchedSpan để so khớp Recall@K cấp Khoản.

    unit_type="khoan" -> khoan_id (case chuẩn, có Khoản thật).
    unit_type="dieu_fallback"/"doc_fallback" -> B0 không có Khoản thật để
    gán, context_id là đơn vị chính xác nhất đã biết -> dùng context_id làm
    "unit đúng" (retrieved cũng phải trả về unit_id trùng context_id ở case
    này để tính là hit, xem retrieve.expand_to_units)."""
    return span["khoan_id"] if span["unit_type"] == "khoan" else span["context_id"]


def recall_at_k_unit(
    labels: dict[str, list[MatchedSpan]],
    retrieved: dict[str, list[dict]],
    k: int,
) -> dict:
    """Recall@K cấp KHOẢN/Dieu — đo sau Expand (B2) hoặc sau Layer 2/3
    (bi-encoder/reranker, khi có). `retrieved` là list[dict] có key
    "unit_id" (vd UnitCandidate) đã lấy sẵn top-K_max, SẮP XẾP GIẢM DẦN
    theo score của layer đang đo (Layer 1 doc_score, hoặc Layer 2/3 score
    sau khi có)."""
    n_no_label = 0
    hits = 0
    n_evaluated = 0
    for qid, spans in labels.items():
        if not spans:
            n_no_label += 1
            continue
        true_units = {_true_unit_id(m) for m in spans}
        retrieved_units = {r["unit_id"] for r in retrieved.get(qid, [])[:k]}
        n_evaluated += 1
        if true_units & retrieved_units:
            hits += 1
    recall = hits / n_evaluated if n_evaluated else 0.0
    return {
        "recall_at_k": recall,
        "k": k,
        "n_evaluated": n_evaluated,
        "n_no_label": n_no_label,
        "n_hits": hits,
    }


def recall_at_multiple_k_unit(
    labels: dict[str, list[MatchedSpan]],
    retrieved: dict[str, list[dict]],
    ks: list[int],
) -> list[dict]:
    """Recall@K cấp Khoản cho nhiều giá trị K — tiện cho bảng so sánh Layer
    1 vs +Layer 2 vs +Layer 3 (bằng chứng cho B5)."""
    return [recall_at_k_unit(labels, retrieved, k) for k in ks]
