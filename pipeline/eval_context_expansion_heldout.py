"""
Context expansion decision (B3) — so sánh 2 biến thể trên CÙNG top-3 Khoản
đã chọn (`select_top_n_khoan`, giống `build_qa_packages_heldout.py`):

  (a) HIỆN TẠI  — context = text của TỪNG KHOẢN trong top-3
  (b) MỞ RỘNG    — context = text của CẢ ĐIỀU chứa từng Khoản trong top-3
                    (loại trùng lặp nếu 2 Khoản top-3 rơi cùng 1 Điều)

Đo Hit rate (gold Khoản có NẰM TRONG context đã trả không — không cần khớp
unit_id chính xác, chỉ cần gold text nằm trong văn bản context) + độ dài
trung bình (đại diện chi phí token cho Generator), có bootstrap CI cho
chênh lệch Hit rate (tái dùng cách làm trong
eval_layer3_citation_labels_heldout.py).

Phương án (b) chọn "cả Điều" (không phải Khoản liền kề) vì khớp đúng tiêu
chí đã dùng đo trần chunking 31,1% (measure_chunking_ceiling.py) và khớp
đề xuất cũ trong src/b3_eval_recall/README.md.

CHỈ ĐO + KHUYẾN NGHỊ — không tự sửa package.py/build_qa_packages_heldout.py.

Chạy: python pipeline/eval_context_expansion_heldout.py
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config, io_utils
from src.b2_retrieval.retrieve import get_unit_text
from src.b4_span_selection.selection import select_top_n_khoan

RERANK_PATH = Path("outputs/layer3/L3_rerank_heldout.jsonl")
B0_LABELS_PATH = config.OUTPUTS_DIR / "b0_labels_train_heldout.json"
CITATION_LABELS_DIR = config.OUTPUTS_DIR
N_BOOTSTRAP = 2000
SEED = 42
# TOP_N=1 (31/08/2026): Generator (C) xác nhận CHỈ nhận đúng 1 context, không xử lý
# nhiều context — đảo ngược quyết định top-3 cũ (13/08). Đo lại trên top-1 vì bối
# cảnh đổi hẳn: không còn top-2/3 dự phòng, "dư" (cả Điều) có thể quan trọng hơn.
TOP_N = 1


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
    merged: dict[str, set[str]] = {}
    for qid, units in b0_gold.items():
        merged.setdefault(qid, set()).update(units)
    for qid, units in citation_gold.items():
        merged.setdefault(qid, set()).update(units)
    return merged


def load_after() -> dict[str, list[dict]]:
    result = {}
    with open(RERANK_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            result[r["qid"]] = r["ranked_units"]
    return result


def khoan_id_to_dieu_id(khoan_id: str) -> str:
    """"{context_id}_{dieu_so}_{khoan_so}" -> "{context_id}_{dieu_so}"."""
    parts = khoan_id.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else khoan_id


def main():
    print("Đang load gold labels + Layer 3 rerank + parsed_corpus...", flush=True)
    heldout_ids = load_heldout_ids()
    b0_gold = load_b0_gold()
    citation_gold = load_citation_gold(heldout_ids)
    gold = merge_gold(b0_gold, citation_gold)
    after = load_after()
    parsed_corpus = io_utils.load_parsed_corpus()

    print(f"Union gold: {len(gold)} câu\n", flush=True)

    hit_a = {}  # qid -> bool (hit với context (a) hiện tại)
    hit_b = {}  # qid -> bool (hit với context (b) mở rộng)
    len_a_list = []
    len_b_list = []

    for qid, gold_units in gold.items():
        ranked_units = after.get(qid, [])
        if not ranked_units:
            continue

        # dùng đúng cơ chế build_qa_packages_heldout.py: khoan_list cần field "text"/"score"
        khoan_list = [
            {"unit_id": u["unit_id"], "context_id": u["context_id"],
             "text": get_unit_text(parsed_corpus, u["unit_id"]), "score": u["score"]}
            for u in ranked_units
        ]
        top_units = select_top_n_khoan("", khoan_list, top_n=TOP_N, already_ranked=True)

        gold_texts = [get_unit_text(parsed_corpus, gu) for gu in gold_units]

        # (a) hiện tại — chỉ Khoản
        context_a_parts = [u["text"] for u in top_units]
        context_a = "\n\n".join(context_a_parts)
        hit_a[qid] = any(gt and gt in context_a for gt in gold_texts)
        len_a_list.append(len(context_a))

        # (b) mở rộng — cả Điều chứa từng Khoản, loại trùng dieu_id
        seen_dieu_ids = set()
        context_b_parts = []
        for u in top_units:
            dieu_id = khoan_id_to_dieu_id(u["unit_id"])
            if dieu_id in seen_dieu_ids:
                continue
            seen_dieu_ids.add(dieu_id)
            dieu_text = get_unit_text(parsed_corpus, dieu_id)
            context_b_parts.append(dieu_text if dieu_text.strip() else u["text"])
        context_b = "\n\n".join(context_b_parts)
        hit_b[qid] = any(gt and gt in context_b for gt in gold_texts)
        len_b_list.append(len(context_b))

    n = len(hit_a)
    rate_a = sum(hit_a.values()) / n
    rate_b = sum(hit_b.values()) / n
    avg_len_a = sum(len_a_list) / n
    avg_len_b = sum(len_b_list) / n

    # bootstrap CI cho chênh lệch hit rate
    qids = list(hit_a.keys())
    rng = random.Random(SEED)
    diffs = []
    for _ in range(N_BOOTSTRAP):
        sample = [qids[rng.randrange(n)] for _ in range(n)]
        ra = sum(hit_a[q] for q in sample) / n
        rb = sum(hit_b[q] for q in sample) / n
        diffs.append(rb - ra)
    diffs.sort()
    lo = diffs[int(0.025 * N_BOOTSTRAP)]
    hi = diffs[int(0.975 * N_BOOTSTRAP)]
    sig = "*** (có ý nghĩa thống kê)" if (lo > 0 or hi < 0) else "(không có ý nghĩa thống kê)"

    print("=" * 90, flush=True)
    print(f"SO SÁNH (n={n} câu, top-{TOP_N} Khoản sau Layer 3)", flush=True)
    print("=" * 90, flush=True)
    print(f"\n{'Biến thể':<30} | {'Hit rate':>10} | {'Độ dài TB (ký tự)':>20}")
    print("-" * 65)
    print(f"{'(a) Hiện tại — chỉ Khoản':<30} | {rate_a:>9.1%} | {avg_len_a:>20,.0f}")
    print(f"{'(b) Mở rộng — cả Điều':<30} | {rate_b:>9.1%} | {avg_len_b:>20,.0f}")
    print("-" * 65)
    print(f"\nChênh lệch Hit rate (b - a): {rate_b - rate_a:+.1%}  95% CI [{lo:+.1%}, {hi:+.1%}] {sig}")
    print(f"Chênh lệch độ dài context: {avg_len_b - avg_len_a:+,.0f} ký tự "
          f"({(avg_len_b / avg_len_a - 1):+.1%})")

    print("\n" + "=" * 90, flush=True)
    if (rate_b - rate_a) > 0 and lo > 0:
        print("KHUYẾN NGHỊ: Hit rate tăng có ý nghĩa thống kê -> CÂN NHẮC áp dụng (b), "
              "nhưng cần xem xét chi phí độ dài context tăng thêm ở trên.", flush=True)
    else:
        print("KHUYẾN NGHỊ: Hit rate KHÔNG tăng có ý nghĩa thống kê -> GIỮ NGUYÊN (a), "
              "không đáng đổi lấy chi phí độ dài context tăng thêm.", flush=True)


if __name__ == "__main__":
    main()
