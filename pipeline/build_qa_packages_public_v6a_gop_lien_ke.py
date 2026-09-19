"""
Build QA packages v6a (top-3, GỘP KHOẢN LIỀN KỀ có ngân sách + dedup chuỗi y hệt)
cho 1.000 câu public-official.json.

V6a = V5 với ĐÚNG 1 THAY ĐỔI: thay "expand cả Điều nếu vừa ngân sách, không thì
giữ nguyên Khoản gốc" (nhị phân, của V5) bằng "mở rộng dần sang Khoản liền kề
CÙNG ĐIỀU (tới/lui quanh Khoản đã chọn) cho tới khi chạm ngân sách" (mịn hơn).
Mọi thứ khác GIỮ NGUYÊN y hệt V5: TOP_N=3 cố định, PER_ITEM_BUDGET_SYLLABLES=300,
dedup theo chuỗi y hệt (KHÔNG dùng Jaccard), KHÔNG đóng gói thích ứng.

Lý do tách riêng khỏi V6 (gộp cả 3 thay đổi): V6 không tách được nguyên nhân
nếu thắng/thua — không biết công/lỗi do gộp Khoản liền kề, dedup Jaccard, hay
đóng gói thích ứng. V6a cô lập ĐÚNG 1 biến (gộp Khoản liền kề) so với V5, để
biết riêng phần này có ăn điểm không trước khi tốn 1 lượt chạy C (~5 giờ/1.000
câu, theo báo cáo audit V5 thật) cho bản gộp cả 3.

Chạy: python pipeline/build_qa_packages_public_v6a_gop_lien_ke.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config, io_utils
from src.b2_retrieval.retrieve import get_unit_text
from src.b6_context_package.package import build_context_item, build_qa_package, write_qa_packages_json

RERANK_PATH = Path("outputs/layer3/L3_rerank_public.jsonl")  # CŨ, giữ cố định — giống hệt V5, so sạch
OUT_PATH_JSON = config.OUTPUTS_DIR / "qa_packages_public_v6a_gop_lien_ke.json"
TOP_N = 3  # giữ nguyên như V5 — KHÔNG đổi sang thích ứng
PER_ITEM_BUDGET_SYLLABLES = 300  # giữ nguyên như V5


def infer_unit_type(unit_id: str, context_id: str) -> str:
    suffix = unit_id[len(context_id) + 1:] if unit_id.startswith(context_id + "_") else ""
    n_parts = len([p for p in suffix.split("_") if p])
    if n_parts >= 2:
        return "khoan"
    if n_parts == 1:
        return "dieu_fallback"
    return "doc_fallback"


def syl_count(text: str) -> int:
    return len(text.split())


def normalize_for_dedup(text: str) -> str:
    """Giống hệt V5 — dedup theo CHUỖI Y HỆT, KHÔNG dùng Jaccard (đó là phần
    riêng của V6, không đưa vào đây để giữ V6a chỉ đổi đúng 1 biến)."""
    return " ".join(text.split()).strip().lower()


def find_dieu_and_index(parsed_corpus: dict, context_id: str, khoan_id: str):
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
    """KHÁC V5 ở đúng chỗ này: gộp Khoản liền kề thay vì nhị phân cả-Điều-hoặc-không."""
    if u["unit_type"] != "khoan":
        return u["text"], False
    return gop_khoan_lien_ke(parsed_corpus, u["context_id"], u["unit_id"], PER_ITEM_BUDGET_SYLLABLES)


def main():
    print("Dang load parsed_corpus + public-official.json + ket qua Layer 3...", flush=True)
    parsed_corpus = io_utils.load_parsed_corpus()
    qa_public = io_utils.load_public_official()

    with open(config.OUTPUTS_DIR / "doc_number_index.json", encoding="utf-8") as f:
        doc_number_index = json.load(f)["usable"]
    context_id_to_doc_number = {v: k for k, v in doc_number_index.items()}

    n_empty_text = 0
    n_merged = 0
    n_deduped_skips = 0
    n_underfilled = 0
    total_syls = []
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
            seen_norm_texts: set[str] = set()
            total_syl = 0
            for u in khoan_list:
                if len(context_items) >= TOP_N:
                    break
                text, merged = decide_text(u, parsed_corpus)
                if not text.strip():
                    n_empty_text += 1
                    continue
                norm = normalize_for_dedup(text)
                if norm in seen_norm_texts:
                    n_deduped_skips += 1
                    continue
                seen_norm_texts.add(norm)
                if merged:
                    n_merged += 1
                doc = parsed_corpus.get(u["context_id"], {})
                context_items.append(build_context_item(
                    context_id=u["context_id"], unit_id=u["unit_id"], unit_type=u["unit_type"],
                    text=text, source_name=doc.get("name", ""), source_link=doc.get("link", ""),
                    document_number=context_id_to_doc_number.get(u["context_id"], ""),
                    retrieval_score=u["score"],
                ))
                total_syl += syl_count(text)

            if len(context_items) < TOP_N:
                n_underfilled += 1
            total_syls.append(total_syl)

            pkg = build_qa_package(qid, question, context_items, reference_answer)
            packages.append(pkg)

    n_written = write_qa_packages_json(packages, OUT_PATH_JSON)
    total_syls.sort()
    n = len(total_syls)
    print(f"Da ghi {n_written} QAPackage -> {OUT_PATH_JSON}", flush=True)
    print(f"Kiem tra: so context rong (bi bo qua) = {n_empty_text}", flush=True)
    print(f"So Khoan da gop lien ke (mo rong ngoai Khoan goc) = {n_merged}", flush=True)
    print(f"So ung vien bi bo qua vi TRUNG chuoi y het = {n_deduped_skips}", flush=True)
    print(f"So cau KHONG du {TOP_N} context duy nhat sau dedup = {n_underfilled}", flush=True)
    print(f"Do dai tong (am tiet): p10={total_syls[n//10]} median={total_syls[n//2]} p90={total_syls[9*n//10]} mean={sum(total_syls)/n:.0f}", flush=True)
    print(f"  <LEN_FLOOR(250)={sum(1 for s in total_syls if s<250)/n:.1%}  >LEN_CAP(1200)={sum(1 for s in total_syls if s>1200)/n:.1%}", flush=True)
    print(f"Kich thuoc file: {OUT_PATH_JSON.stat().st_size / 1024:.1f} KB", flush=True)


if __name__ == "__main__":
    main()
