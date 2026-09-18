"""
Kiem tra kha nang noi `data123` (citation_metadata do C trich tu reference_answer,
KHONG phai context_id that) vao corpus that cua B, va doi chieu voi nhan B0 hien co.

Chi la SCRIPT KIEM CHUNG (khong sua production) — tra loi 3 cau hoi:
1. document_number trich boi C co tra duoc context_id that qua doc_number_index.json khong?
2. Trong context_id do, co tim duoc dung Dieu/Khoan theo so da trich khong?
3. Ket qua noi duoc co KHOP voi nhan B0 dang co (cho cau trung ca 2 nguon) khong?

Chay: python pipeline/eval_data123_link_check.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config, io_utils

DATA123_DIR = config.REPO_ROOT / "data123" / "data"
HELDOUT_LABELS_PATH = config.OUTPUTS_DIR / "b0_labels_train_heldout.json"

ARTICLE_NUM_RE = re.compile(r"(\d+[a-zA-Z]?)")
CLAUSE_NUM_RE = re.compile(r"(\d+)")


def load_data123_records() -> dict[str, dict]:
    """{qid: item} tu ca 3 split, item co oracle_context.citation_metadata."""
    records = {}
    for split in ["train", "validation", "test"]:
        path = DATA123_DIR / split / "baseline_eligible.json"
        for item in json.loads(path.read_text(encoding="utf-8")):
            records[str(item["id"])] = item
    return records


def extract_first_num(text: str, pattern: re.Pattern) -> str | None:
    m = pattern.search(text)
    return m.group(1) if m else None


def find_khoan_id(parsed_corpus: dict, context_id: str, dieu_so: str, khoan_so: str | None) -> tuple[str | None, str]:
    """Tra khoan_id/dieu_id that trong parsed_corpus theo so da trich.
    Tra (unit_id_or_None, ly_do_neu_that_bai)."""
    doc = parsed_corpus.get(context_id)
    if doc is None:
        return None, "context_id khong co trong parsed_corpus"
    for dieu in doc.get("dieu", []):
        d_id = dieu.get("dieu_id", "")
        d_so = d_id.rsplit("_", 1)[-1] if d_id else ""
        if d_so != dieu_so:
            continue
        if khoan_so is None or not dieu.get("khoan"):
            return d_id, "ok_dieu_fallback"
        for k in dieu["khoan"]:
            k_so = k["khoan_id"].rsplit("_", 1)[-1]
            if k_so == khoan_so:
                return k["khoan_id"], "ok_khoan"
        return None, f"khong tim thay khoan {khoan_so} trong dieu {dieu_so}"
    return None, f"khong tim thay dieu {dieu_so} trong context_id"


def main():
    print("Dang load parsed_corpus + doc_number_index + nhan B0 + data123...", flush=True)
    parsed_corpus = io_utils.load_parsed_corpus()
    doc_number_index = json.loads((config.OUTPUTS_DIR / "doc_number_index.json").read_text(encoding="utf-8"))["usable"]
    doc_number_to_context = {k: v for k, v in doc_number_index.items()}

    b0_data = json.loads(HELDOUT_LABELS_PATH.read_text(encoding="utf-8"))
    b0_labels = b0_data["labels"]
    heldout_ids = set(b0_data["heldout_ids"])

    data123_records = load_data123_records()
    overlap_ids = heldout_ids & set(data123_records.keys())
    print(f"So cau heldout co trong data123: {len(overlap_ids)}/{len(heldout_ids)}", flush=True)

    n_no_docnum = 0
    n_docnum_not_found = 0
    n_no_article = 0
    n_dieu_khoan_not_found = 0
    n_linked_ok = 0
    n_agree_b0 = 0
    n_disagree_b0 = 0
    n_no_b0_label = 0
    disagree_examples = []

    for qid in sorted(overlap_ids):
        item = data123_records[qid]
        cm = item["oracle_context"]["citation_metadata"]
        doc_numbers = cm.get("document_numbers") or []
        articles = cm.get("articles") or []
        clauses = cm.get("clauses") or []

        if not doc_numbers:
            n_no_docnum += 1
            continue
        context_id = doc_number_to_context.get(doc_numbers[0])
        if not context_id:
            n_docnum_not_found += 1
            continue
        if not articles:
            n_no_article += 1
            continue
        dieu_so = extract_first_num(articles[0], ARTICLE_NUM_RE)
        khoan_so = extract_first_num(clauses[0], CLAUSE_NUM_RE) if clauses else None

        unit_id, reason = find_khoan_id(parsed_corpus, context_id, dieu_so, khoan_so)
        if unit_id is None:
            n_dieu_khoan_not_found += 1
            continue

        n_linked_ok += 1

        b0_spans = b0_labels.get(qid, [])
        b0_khoan_ids = {s["khoan_id"] for s in b0_spans if s["unit_type"] == "khoan" and s["confidence"] >= config.B0_CONFIDENCE_TRUST}
        if not b0_khoan_ids:
            n_no_b0_label += 1
        else:
            # so cong bang: cung DIEU la khop (data123 co the chi dung toi cap Dieu
            # neu thieu so Khoan cu the), chi tinh LECH khi khac Dieu hoac khac Khoan
            # ro rang (ca 2 phia deu co so Khoan nhung khac nhau).
            unit_dieu_prefix = "_".join(unit_id.split("_")[:2])  # context_id_dieu_so
            b0_dieu_prefixes = {"_".join(b.split("_")[:2]) for b in b0_khoan_ids}
            if unit_id in b0_khoan_ids or unit_dieu_prefix in b0_dieu_prefixes:
                n_agree_b0 += 1
            else:
                n_disagree_b0 += 1
                if len(disagree_examples) < 5:
                    disagree_examples.append((qid, unit_id, b0_khoan_ids))

    print(f"\n=== KET QUA NOI (n={len(overlap_ids)}) ===")
    print(f"Thieu document_number: {n_no_docnum}")
    print(f"document_number khong tra duoc context_id: {n_docnum_not_found}")
    print(f"Thieu article: {n_no_article}")
    print(f"Khong tim thay Dieu/Khoan dung so trong corpus: {n_dieu_khoan_not_found}")
    print(f"NOI THANH CONG: {n_linked_ok} ({n_linked_ok/len(overlap_ids):.1%})")

    print(f"\n=== DOI CHIEU VOI NHAN B0 (trong so {n_linked_ok} noi thanh cong) ===")
    print(f"B0 khong co nhan tin cay o cau nay: {n_no_b0_label}")
    print(f"KHOP voi B0: {n_agree_b0}")
    print(f"LECH voi B0: {n_disagree_b0}")
    if n_agree_b0 + n_disagree_b0 > 0:
        print(f"Ty le khop (trong so co ca 2 nhan): {n_agree_b0/(n_agree_b0+n_disagree_b0):.1%}")

    if disagree_examples:
        print("\nVi du lech (toi da 5):")
        for qid, unit_id, b0_ids in disagree_examples:
            print(f"  qid={qid}  data123->{unit_id}  B0->{b0_ids}")


if __name__ == "__main__":
    main()
