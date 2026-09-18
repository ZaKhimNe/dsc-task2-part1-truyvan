"""
Build QA packages v6 (đóng gói thích ứng + dedup Jaccard + gộp Khoản liền kề)
cho 1.000 câu public-official.json.

DÙNG NGUYÊN `outputs/layer3/L3_rerank_public.jsonl` (reranker CŨ, bi-encoder CŨ)
— GIỮ NGUYÊN ranking để so sạch với V5 (chỉ đổi selector, không đổi ranking).
So sánh với model retrieval mới là việc của "V5b", làm riêng — không trộn 2 biến.

V6 sửa 3 lỗ đã đo được của V5 (retrieval/docs/BAO_CAO_V1_DEN_V5.md, PLAN_KE_THUA_TEAM_IR.md):

1. ĐỘ DÀI THÍCH ỨNG (thay "luôn đúng 3 context" của V5): duyệt ranked_units theo
   thứ tự điểm, dừng khi tổng đã đạt TARGET_TOTAL_SYLLABLES (~535 âm tiết context
   = 620 tối ưu đo được cho câu trả lời − 84 overhead Lead/Conclusion đo thật từ
   answers.json), tối đa MAX_CONTEXTS=6. V5 luôn lấy đúng 3 dù đã đủ hay chưa đủ
   ngân sách -> độ dài trải rất rộng (p10=267, p90=945 âm tiết).

2. DEDUP JACCARD (thay dedup CHUỖI Y HỆT của V5): 2 context trùng ý nhưng khác vài
   chữ (vd 2 bản cũ/mới cùng 1 điều luật) sẽ bị bỏ qua nếu overlap token >0,7 —
   V5 chỉ bắt được trùng chuỗi y hệt, còn sót 12,8% số câu có cặp gần-trùng.

3. GỘP KHOẢN LIỀN KỀ (thay "expand cả Điều nếu vừa ngân sách, không thì giữ
   nguyên Khoản gốc" nhị phân của V5): mở rộng dần sang Khoản liền kề CÙNG ĐIỀU
   (tới/lui quanh Khoản đã chọn) cho tới khi chạm PER_ITEM_BUDGET_SYLLABLES —
   mịn hơn V5 (V5 mất sạch phần mở rộng nếu cả Điều hơi dài quá ngân sách một
   chút). Corroborate với R6.1 của Team IR (gộp đoạn liền kề tới ~1.800 ký tự) —
   nhưng đây là phát hiện của B (V2 vs V3 trên LB thật), không phải kế thừa từ
   họ — xem PLAN_KE_THUA_TEAM_IR.md Phase 1 để biết vì sao khác quy mô.

`article`/`clause` trong metadata LUÔN ghi Khoản GỐC được chọn (không phải range
đã gộp) — giữ đúng quy ước cũ (xem comment trong package.py / v2 script).

Chạy: python pipeline/build_qa_packages_public_v6_adaptive_dedup.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config, io_utils
from src.b2_retrieval.retrieve import get_unit_text
from src.b6_context_package.package import build_context_item, build_qa_package, write_qa_packages_json

RERANK_PATH = Path("outputs/layer3/L3_rerank_public.jsonl")  # CŨ, giữ cố định để so sạch với V5
OUT_PATH_JSON = config.OUTPUTS_DIR / "qa_packages_public_v6_adaptive_dedup.json"

TARGET_TOTAL_SYLLABLES = 535  # 620 (toi uu do duoc cho cau tra loi) - 84 (overhead Lead+Conclusion, do that)
PER_ITEM_BUDGET_SYLLABLES = 250  # ngan sach gop Khoan lien ke cho MOI context
MAX_CONTEXTS = 6
LEN_CAP_SYLLABLES = 1200  # da do tren dev_fast (eval_harness/baseline_template.py)
JACCARD_DEDUP_THRESHOLD = 0.7


def infer_unit_type(unit_id: str, context_id: str) -> str:
    suffix = unit_id[len(context_id) + 1 :] if unit_id.startswith(context_id + "_") else ""
    n_parts = len([p for p in suffix.split("_") if p])
    if n_parts >= 2:
        return "khoan"
    if n_parts == 1:
        return "dieu_fallback"
    return "doc_fallback"


def syl_count(text: str) -> int:
    return len(text.split())


def normalize_tokens(text: str) -> set[str]:
    return set(" ".join(text.split()).lower().split())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_dieu_and_index(parsed_corpus: dict, context_id: str, khoan_id: str):
    """Tra ve (danh_sach_khoan_cua_dieu_chua_no, vi_tri_trong_danh_sach) hoac
    (None, None) neu khong tim thay (an toan, khong crash)."""
    doc = parsed_corpus.get(context_id)
    if not doc:
        return None, None
    for dieu in doc.get("dieu", []):
        khoan_list = dieu.get("khoan", [])
        for i, k in enumerate(khoan_list):
            if k["khoan_id"] == khoan_id:
                return khoan_list, i
    return None, None


def gop_khoan_lien_ke(parsed_corpus: dict, context_id: str, khoan_id: str, budget: int) -> tuple[str, bool]:
    """Mo rong tu Khoan da chon sang Khoan lien ke CUNG DIEU (toi/lui) cho toi
    khi cham ngan sach. Tra (text_da_ghep, co_mo_rong_khong)."""
    khoan_list, idx = find_dieu_and_index(parsed_corpus, context_id, khoan_id)
    if khoan_list is None:
        return get_unit_text(parsed_corpus, khoan_id), False

    lo = hi = idx
    total = syl_count(khoan_list[idx]["text"])
    while total < budget:
        can_hi = hi + 1 < len(khoan_list)
        can_lo = lo - 1 >= 0
        if not can_hi and not can_lo:
            break
        if can_hi:
            hi += 1
            total += syl_count(khoan_list[hi]["text"])
        elif can_lo:
            lo -= 1
            total += syl_count(khoan_list[lo]["text"])
    merged = (lo, hi) != (idx, idx)
    text = "\n\n".join(khoan_list[i]["text"] for i in range(lo, hi + 1))
    return text, merged


def decide_text(u: dict, parsed_corpus: dict) -> tuple[str, bool]:
    if u["unit_type"] != "khoan":
        return u["text"], False
    return gop_khoan_lien_ke(parsed_corpus, u["context_id"], u["unit_id"], PER_ITEM_BUDGET_SYLLABLES)


def main():
    print("Dang load parsed_corpus + public-official.json + ket qua Layer 3 (CU, co dinh)...", flush=True)
    parsed_corpus = io_utils.load_parsed_corpus()
    qa_public = io_utils.load_public_official()

    with open(config.OUTPUTS_DIR / "doc_number_index.json", encoding="utf-8") as f:
        doc_number_index = json.load(f)["usable"]
    context_id_to_doc_number = {v: k for k, v in doc_number_index.items()}

    n_empty_text = 0
    n_merged_khoan = 0
    n_deduped_skips = 0
    n_underfilled = 0
    context_counts: dict[int, int] = {}
    total_syls: list[int] = []
    packages = []

    with open(RERANK_PATH, encoding="utf-8") as f_in:
        for line in f_in:
            r = json.loads(line)
            qid = r["qid"]
            question = qa_public[qid]["question"]
            reference_answer = qa_public[qid].get("answer")

            khoan_list = []
            for u in r["ranked_units"]:
                text = get_unit_text(parsed_corpus, u["unit_id"])
                khoan_list.append({
                    "unit_id": u["unit_id"],
                    "context_id": u["context_id"],
                    "text": text,
                    "score": u["score"],
                    "unit_type": infer_unit_type(u["unit_id"], u["context_id"]),
                })

            context_items = []
            seen_token_sets: list[set[str]] = []
            total_syl = 0
            for u in khoan_list:
                if len(context_items) >= MAX_CONTEXTS:
                    break
                if total_syl >= TARGET_TOTAL_SYLLABLES and len(context_items) >= 1:
                    break

                text, merged = decide_text(u, parsed_corpus)
                if not text.strip():
                    n_empty_text += 1
                    continue

                tokens = normalize_tokens(text)
                if any(jaccard(tokens, seen) > JACCARD_DEDUP_THRESHOLD for seen in seen_token_sets):
                    n_deduped_skips += 1
                    continue

                n = syl_count(text)
                if context_items and total_syl + n > LEN_CAP_SYLLABLES:
                    break

                if merged:
                    n_merged_khoan += 1
                doc = parsed_corpus.get(u["context_id"], {})
                context_items.append(build_context_item(
                    context_id=u["context_id"], unit_id=u["unit_id"], unit_type=u["unit_type"],
                    text=text, source_name=doc.get("name", ""), source_link=doc.get("link", ""),
                    document_number=context_id_to_doc_number.get(u["context_id"], ""),
                    retrieval_score=u["score"],
                ))
                seen_token_sets.append(tokens)
                total_syl += n

            if len(context_items) < 1:
                n_underfilled += 1
            context_counts[len(context_items)] = context_counts.get(len(context_items), 0) + 1
            total_syls.append(total_syl)

            pkg = build_qa_package(qid, question, context_items, reference_answer)
            packages.append(pkg)

    n_written = write_qa_packages_json(packages, OUT_PATH_JSON)
    total_syls.sort()
    n = len(total_syls)
    print(f"Da ghi {n_written} QAPackage -> {OUT_PATH_JSON}", flush=True)
    print(f"Kiem tra: so context rong (bi bo qua) = {n_empty_text}", flush=True)
    print(f"So Khoan da gop lien ke (mo rong ngoai Khoan goc) = {n_merged_khoan}", flush=True)
    print(f"So ung vien bi bo qua vi TRUNG (Jaccard>{JACCARD_DEDUP_THRESHOLD}) = {n_deduped_skips}", flush=True)
    print(f"So cau KHONG co context nao (underfilled) = {n_underfilled}", flush=True)
    print(f"Phan bo so context/cau: {dict(sorted(context_counts.items()))}", flush=True)
    print(f"Do dai tong (am tiet): p10={total_syls[n//10]} median={total_syls[n//2]} p90={total_syls[9*n//10]} mean={sum(total_syls)/n:.0f}", flush=True)
    print(f"  <LEN_FLOOR(250)={sum(1 for s in total_syls if s<250)/n:.1%}  >LEN_CAP(1200)={sum(1 for s in total_syls if s>1200)/n:.1%}", flush=True)
    print(f"Kich thuoc file: {OUT_PATH_JSON.stat().st_size / 1024:.1f} KB", flush=True)


if __name__ == "__main__":
    main()
