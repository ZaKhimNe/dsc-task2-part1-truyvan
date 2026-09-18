"""
Xay bo nhan gop (B0 + data123-link) cho 800 cau heldout — lam giau ground
truth dung cho moi phep do Hit@K/recall (Phase 2a, 2b, oracle...) da lam tu
truoc, deu chi dang dua tren ~198 cau B0 tin cay.

Logic gop (bao thu, khong tu y "cuu" ca lech):
  - Ca 2 nguon THONG NHAT (cung Dieu tro len)  -> "agree"    (tin cay CAO NHAT)
  - Chi B0 co nhan tin cay                      -> "b0_only"  (nhu truoc gio)
  - Chi data123 noi duoc, B0 khong co nhan       -> "d123_only" (MOI, truoc day khong co)
  - Ca 2 co nhan nhung LECH nhau                 -> "disagree" (KHONG dua vao tap tin cay,
                                                     giu lai rieng de doc tay sau)

Tap "tin cay" dung cho do luong = agree + b0_only + d123_only (bo disagree).

Chay: python pipeline/build_gold_labels_merged_heldout.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config, io_utils

DATA123_DIR = config.REPO_ROOT / "data123" / "data"
HELDOUT_LABELS_PATH = config.OUTPUTS_DIR / "b0_labels_train_heldout.json"
OUT_PATH = config.OUTPUTS_DIR / "gold_labels_merged_heldout.json"

ARTICLE_NUM_RE = re.compile(r"(\d+[a-zA-Z]?)")
CLAUSE_NUM_RE = re.compile(r"(\d+)")


def load_data123_records() -> dict[str, dict]:
    records = {}
    for split in ["train", "validation", "test"]:
        path = DATA123_DIR / split / "baseline_eligible.json"
        for item in json.loads(path.read_text(encoding="utf-8")):
            records[str(item["id"])] = item
    return records


def extract_first_num(text: str, pattern: re.Pattern) -> str | None:
    m = pattern.search(text)
    return m.group(1) if m else None


def find_unit_id(parsed_corpus: dict, context_id: str, dieu_so: str, khoan_so: str | None) -> str | None:
    doc = parsed_corpus.get(context_id)
    if doc is None:
        return None
    for dieu in doc.get("dieu", []):
        d_id = dieu.get("dieu_id", "")
        d_so = d_id.rsplit("_", 1)[-1] if d_id else ""
        if d_so != dieu_so:
            continue
        if khoan_so is None or not dieu.get("khoan"):
            return d_id
        for k in dieu["khoan"]:
            if k["khoan_id"].rsplit("_", 1)[-1] == khoan_so:
                return k["khoan_id"]
        return None
    return None


def link_data123(item: dict, parsed_corpus: dict, doc_number_to_context: dict) -> str | None:
    cm = item["oracle_context"]["citation_metadata"]
    doc_numbers = cm.get("document_numbers") or []
    articles = cm.get("articles") or []
    clauses = cm.get("clauses") or []
    if not doc_numbers or not articles:
        return None
    context_id = doc_number_to_context.get(doc_numbers[0])
    if not context_id:
        return None
    dieu_so = extract_first_num(articles[0], ARTICLE_NUM_RE)
    if not dieu_so:
        return None
    khoan_so = extract_first_num(clauses[0], CLAUSE_NUM_RE) if clauses else None
    return find_unit_id(parsed_corpus, context_id, dieu_so, khoan_so)


def dieu_prefix(unit_id: str) -> str:
    return "_".join(unit_id.split("_")[:2])


def main():
    print("Dang load parsed_corpus + doc_number_index + B0 + data123...", flush=True)
    parsed_corpus = io_utils.load_parsed_corpus()
    doc_number_index = json.loads((config.OUTPUTS_DIR / "doc_number_index.json").read_text(encoding="utf-8"))["usable"]
    doc_number_to_context = dict(doc_number_index)

    b0_data = json.loads(HELDOUT_LABELS_PATH.read_text(encoding="utf-8"))
    b0_labels = b0_data["labels"]
    heldout_ids = b0_data["heldout_ids"]

    data123_records = load_data123_records()

    merged = {}
    counts = {"agree": 0, "b0_only": 0, "d123_only": 0, "disagree": 0, "none": 0}

    for qid in heldout_ids:
        b0_spans = b0_labels.get(qid, [])
        b0_khoan_ids = {
            s["khoan_id"] for s in b0_spans
            if s["unit_type"] == "khoan" and s["confidence"] >= config.B0_CONFIDENCE_TRUST
        }

        d123_unit_id = None
        if qid in data123_records:
            d123_unit_id = link_data123(data123_records[qid], parsed_corpus, doc_number_to_context)

        if b0_khoan_ids and d123_unit_id:
            b0_prefixes = {dieu_prefix(u) for u in b0_khoan_ids}
            if d123_unit_id in b0_khoan_ids or dieu_prefix(d123_unit_id) in b0_prefixes:
                merged[qid] = {"source": "agree", "khoan_ids": sorted(b0_khoan_ids | {d123_unit_id})}
                counts["agree"] += 1
            else:
                merged[qid] = {
                    "source": "disagree",
                    "b0_khoan_ids": sorted(b0_khoan_ids),
                    "d123_unit_id": d123_unit_id,
                }
                counts["disagree"] += 1
        elif b0_khoan_ids:
            merged[qid] = {"source": "b0_only", "khoan_ids": sorted(b0_khoan_ids)}
            counts["b0_only"] += 1
        elif d123_unit_id:
            merged[qid] = {"source": "d123_only", "khoan_ids": [d123_unit_id]}
            counts["d123_only"] += 1
        else:
            counts["none"] += 1

    OUT_PATH.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    n_trusted = counts["agree"] + counts["b0_only"] + counts["d123_only"]
    print(f"\nDa ghi -> {OUT_PATH}")
    print(f"Tong so cau heldout: {len(heldout_ids)}")
    print(f"  agree (ca 2 nguon khop)      : {counts['agree']}")
    print(f"  b0_only (chi B0 co)          : {counts['b0_only']}")
    print(f"  d123_only (chi data123 co)   : {counts['d123_only']}")
    print(f"  disagree (loai khoi tap tin cay): {counts['disagree']}")
    print(f"  none (khong nguon nao co)    : {counts['none']}")
    print(f"\n=> TAP TIN CAY MOI: {n_trusted} cau (so voi {sum(1 for s in b0_labels.values() if any(x['unit_type']=='khoan' and x['confidence']>=config.B0_CONFIDENCE_TRUST for x in s))} cau chi dung B0 truoc day)")


if __name__ == "__main__":
    main()
