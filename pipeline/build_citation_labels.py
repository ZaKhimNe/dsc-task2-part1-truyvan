"""
Match `citation_metadata` (số hiệu văn bản + Điều + Khoản, đã parse sẵn từ
câu trích dẫn trong `data123/`) với corpus của mình qua index số hiệu văn
bản (`build_doc_number_index.py`) — tạo nhãn Khoản-level ĐÁNG TIN CẬY hơn B0
nhiều lần (khớp số hiệu chính xác, không suy đoán mờ).

Chỉ xử lý case CHẮC CHẮN: đúng 1 document_number, đúng 1 article, tối đa 1
clause — case nhiều/thiếu thông tin BỎ QUA, không đoán bừa (đúng tinh thần
"chỉ tin cái chắc chắn" đã áp dụng xuyên suốt dự án).

3 loại kết quả, KHÔNG trộn lẫn:
  - "khoan"      : khớp đủ context_id + Điều + Khoản, unit tồn tại thật
  - "dieu"        : khớp context_id + Điều (không có/không dùng Khoản)
  - "not_found"   : khớp được số hiệu văn bản nhưng Điều/Khoản không tồn tại
                     trong parsed_corpus (có thể do B1 parse lỗi) — ghi nhận
                     riêng để điều tra, KHÔNG tính là nhãn.

Chạy: python pipeline/build_citation_labels.py
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config, io_utils
from src.b2_retrieval.retrieve import get_unit_text

DATA123_DIR = config.DATA123_DIR
DOC_INDEX_PATH = config.OUTPUTS_DIR / "doc_number_index.json"
OUT_DIR = config.OUTPUTS_DIR
SPLITS = ["train", "validation", "test"]

_NUM_RE = re.compile(r"\d+")


def normalize_doc_number(s: str) -> str:
    s = re.sub(r"\s+", "", s)
    return unicodedata.normalize("NFC", s).upper()


def extract_number(s: str) -> str | None:
    m = _NUM_RE.search(s)
    return m.group(0) if m else None


def load_tuning_status() -> tuple[set[str], set[str]]:
    """qid nào đã dùng tune (dev-sample 1.500) vs đã ở tập giữ kín B0 (800)."""
    with open(config.OUTPUTS_DIR / "b0_labels_train_sample.json", encoding="utf-8") as f:
        tuned_ids = set(json.load(f)["sample_ids"])
    with open(config.OUTPUTS_DIR / "b0_labels_train_heldout.json", encoding="utf-8") as f:
        heldout_ids = set(json.load(f)["heldout_ids"])
    return tuned_ids, heldout_ids


def main():
    with open(DOC_INDEX_PATH, encoding="utf-8") as f:
        doc_index = json.load(f)["usable"]
    parsed_corpus = io_utils.load_parsed_corpus()
    tuned_ids, heldout_ids = load_tuning_status()

    for split in SPLITS:
        path = DATA123_DIR / split / "baseline_eligible.json"
        with open(path, encoding="utf-8") as f:
            records = json.load(f)

        stats = {"khoan": 0, "dieu": 0, "not_found": 0, "skipped_ambiguous_citation": 0}
        out_records = []

        for r in records:
            qid = r["id"]
            cm = r["oracle_context"]["citation_metadata"]
            docs = cm["document_numbers"]
            articles = cm["articles"]
            clauses = cm["clauses"]

            if len(docs) != 1 or len(articles) != 1 or len(clauses) > 1:
                stats["skipped_ambiguous_citation"] += 1
                continue

            doc_key = normalize_doc_number(docs[0])
            context_id = doc_index.get(doc_key)
            if context_id is None:
                stats["skipped_ambiguous_citation"] += 1
                continue

            dieu_so = extract_number(articles[0])
            khoan_so = extract_number(clauses[0]) if clauses else None
            if dieu_so is None:
                stats["skipped_ambiguous_citation"] += 1
                continue

            dieu_id = f"{context_id}_{dieu_so}"
            if khoan_so:
                khoan_id = f"{dieu_id}_{khoan_so}"
                text = get_unit_text(parsed_corpus, khoan_id)
                if text.strip():
                    stats["khoan"] += 1
                    out_records.append({
                        "qid": qid, "context_id": context_id, "unit_id": khoan_id,
                        "unit_type": "khoan", "method": "citation_match",
                        "source_document_number": docs[0],
                        "in_tuned_1500": qid in tuned_ids, "in_heldout_800": qid in heldout_ids,
                    })
                    continue
                # Khoan khong ton tai -> thu fallback ve Dieu truoc khi bo cuoc
                text = get_unit_text(parsed_corpus, dieu_id)
                if text.strip():
                    stats["not_found"] += 1  # ghi nhan rieng: khop van ban+Dieu nhung sai Khoan
                    continue
                stats["not_found"] += 1
                continue
            else:
                text = get_unit_text(parsed_corpus, dieu_id)
                if text.strip():
                    stats["dieu"] += 1
                    out_records.append({
                        "qid": qid, "context_id": context_id, "unit_id": dieu_id,
                        "unit_type": "dieu", "method": "citation_match",
                        "source_document_number": docs[0],
                        "in_tuned_1500": qid in tuned_ids, "in_heldout_800": qid in heldout_ids,
                    })
                    continue
                stats["not_found"] += 1

        out_path = OUT_DIR / f"citation_labels_{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_records, f, ensure_ascii=False, indent=2)

        n_total = len(records)
        print(f"=== {split} ({n_total} câu baseline_eligible) ===", flush=True)
        print(f"  Nhãn cấp KHOẢN: {stats['khoan']}", flush=True)
        print(f"  Nhãn cấp ĐIỀU (không có Khoản): {stats['dieu']}", flush=True)
        print(f"  Khớp văn bản nhưng Điều/Khoản không tồn tại (not_found): {stats['not_found']}", flush=True)
        print(f"  Bỏ qua (citation không đủ chắc chắn / văn bản không match): {stats['skipped_ambiguous_citation']}", flush=True)
        print(f"  -> Đã lưu {out_path}\n", flush=True)


if __name__ == "__main__":
    main()
