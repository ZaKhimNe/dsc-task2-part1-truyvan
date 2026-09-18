"""
Build QA packages v6b (ĐÓNG GÓI THÍCH ỨNG, giữ nguyên cách mở rộng nhị phân +
dedup chuỗi y hệt như V5) cho 1.000 câu public-official.json.

V6b = V5 với ĐÚNG 1 THAY ĐỔI: thay "luôn lấy đúng 3 context" (TOP_N cố định
của V5) bằng "duyệt ranked_units theo điểm, dừng khi tổng đã đạt
TARGET_TOTAL_SYLLABLES (hoặc tối đa MAX_CONTEXTS), không cố định số lượng".
Mọi thứ khác GIỮ NGUYÊN y hệt V5: cách mở rộng Khoản->Điều NHỊ PHÂN (expand
cả Điều nếu vừa PER_ITEM_BUDGET_SYLLABLES=300, không thì giữ Khoản gốc), dedup
theo CHUỖI Y HỆT (không dùng Jaccard).

Đây là 1 trong 2 bản tách từ V6 (3 thay đổi gộp chung), cùng với V6a2 (chỉ đổi
gộp Khoản liền kề, độ dài đã kiểm soát ≈ V5) — để biết riêng "số context thay
đổi theo câu" có ăn điểm không, tách khỏi 2 thay đổi kia.

Chạy: python pipeline/build_qa_packages_public_v6b_thich_ung.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config, io_utils
from src.b2_retrieval.retrieve import get_unit_text
from src.b6_context_package.package import build_context_item, build_qa_package, write_qa_packages_json

RERANK_PATH = Path("outputs/layer3/L3_rerank_public.jsonl")  # CŨ, giữ cố định để so sạch với V5
OUT_PATH_JSON = config.OUTPUTS_DIR / "qa_packages_public_v6b_thich_ung.json"

TARGET_TOTAL_SYLLABLES = 535  # giống V6 — 620 (toi uu) - 84 (overhead Lead+Conclusion)
PER_ITEM_BUDGET_SYLLABLES = 300  # GIONG HET V5 (nhi phan, khong phai gop lien ke)
MAX_CONTEXTS = 6
LEN_CAP_SYLLABLES = 1200


def khoan_id_to_dieu_id(khoan_id: str) -> str:
    parts = khoan_id.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else khoan_id


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
    """GIỐNG HỆT V5 — dedup theo CHUỖI Y HỆT, không dùng Jaccard (đó là phần
    riêng của V6, không đưa vào đây để V6b chỉ đổi đúng 1 biến)."""
    return " ".join(text.split()).strip().lower()


def decide_text(u: dict, parsed_corpus: dict) -> tuple[str, bool]:
    """GIỐNG HỆT V5 — nhị phân: expand cả Điều nếu vừa ngân sách, không thì
    giữ nguyên Khoản gốc."""
    raw_text = u["text"]
    if u["unit_type"] != "khoan":
        return raw_text, False
    dieu_text = get_unit_text(parsed_corpus, khoan_id_to_dieu_id(u["unit_id"]))
    if dieu_text.strip() and syl_count(dieu_text) <= PER_ITEM_BUDGET_SYLLABLES:
        return dieu_text, True
    return raw_text, False


def main():
    print("Dang load parsed_corpus + public-official.json + ket qua Layer 3 (CU, co dinh)...", flush=True)
    parsed_corpus = io_utils.load_parsed_corpus()
    qa_public = io_utils.load_public_official()

    with open(config.OUTPUTS_DIR / "doc_number_index.json", encoding="utf-8") as f:
        doc_number_index = json.load(f)["usable"]
    context_id_to_doc_number = {v: k for k, v in doc_number_index.items()}

    n_empty_text = 0
    n_expanded = 0
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

            # KHÁC V5 ở đây: KHÔNG dừng ở TOP_N=3 cố định — duyệt tới khi đạt
            # TARGET_TOTAL_SYLLABLES (hoặc MAX_CONTEXTS), giống cơ chế dừng của V6.
            context_items = []
            seen_norm_texts: set[str] = set()
            total_syl = 0
            for u in khoan_list:
                if len(context_items) >= MAX_CONTEXTS:
                    break
                if total_syl >= TARGET_TOTAL_SYLLABLES and len(context_items) >= 1:
                    break

                text, expanded = decide_text(u, parsed_corpus)
                if not text.strip():
                    n_empty_text += 1
                    continue

                norm = normalize_for_dedup(text)
                if norm in seen_norm_texts:
                    n_deduped_skips += 1
                    continue

                n = syl_count(text)
                if context_items and total_syl + n > LEN_CAP_SYLLABLES:
                    break

                if expanded:
                    n_expanded += 1
                seen_norm_texts.add(norm)
                doc = parsed_corpus.get(u["context_id"], {})
                context_items.append(build_context_item(
                    context_id=u["context_id"], unit_id=u["unit_id"], unit_type=u["unit_type"],
                    text=text, source_name=doc.get("name", ""), source_link=doc.get("link", ""),
                    document_number=context_id_to_doc_number.get(u["context_id"], ""),
                    retrieval_score=u["score"],
                ))
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
    print(f"So context da expand Khoan->Dieu (nhi phan, giong V5) = {n_expanded}", flush=True)
    print(f"So ung vien bi bo qua vi TRUNG chuoi y het = {n_deduped_skips}", flush=True)
    print(f"So cau KHONG co context nao (underfilled) = {n_underfilled}", flush=True)
    print(f"Phan bo so context/cau: {dict(sorted(context_counts.items()))}", flush=True)
    print(f"Do dai tong (am tiet): p10={total_syls[n//10]} median={total_syls[n//2]} p90={total_syls[9*n//10]} mean={sum(total_syls)/n:.0f}", flush=True)
    print(f"  <LEN_FLOOR(250)={sum(1 for s in total_syls if s<250)/n:.1%}  >LEN_CAP(1200)={sum(1 for s in total_syls if s>1200)/n:.1%}", flush=True)
    print(f"Kich thuoc file: {OUT_PATH_JSON.stat().st_size / 1024:.1f} KB", flush=True)


if __name__ == "__main__":
    main()
