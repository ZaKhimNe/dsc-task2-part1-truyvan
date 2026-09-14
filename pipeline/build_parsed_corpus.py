"""
Chạy B1 (parse_document) trên toàn bộ corpus, ghi ra data/parsed_corpus.jsonl.

Mỗi dòng = 1 văn bản: {context_id, name, link, parse_status, dieu, phu_luc_raw}.
Văn bản passage rỗng (20 văn bản đã biết) bị io_utils.iter_corpus bỏ qua từ đầu.

Chạy: python experiments/build_parsed_corpus.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config, io_utils
from src.b1_parser import parser

OUT_PATH = config.DATA_DIR / "parsed_corpus.jsonl"


def main():
    n_matched, n_fallback, n_total = 0, 0, 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for doc in io_utils.iter_corpus():
            context_id = str(doc["id"])
            parsed = parser.parse_document(context_id, doc["passage"])
            n_total += 1
            n_matched += parsed["parse_status"] == "matched"
            n_fallback += parsed["parse_status"] == "fallback"
            record = {
                "context_id": context_id,
                "name": doc["name"],
                "link": doc["link"],
                "parse_status": parsed["parse_status"],
                "dieu": parsed["dieu"],
                "phu_luc_raw": parsed["phu_luc_raw"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Ghi {n_total} văn bản -> {OUT_PATH}")
    print(f"  matched={n_matched} ({n_matched / n_total * 100:.1f}%), "
          f"fallback={n_fallback} ({n_fallback / n_total * 100:.1f}%)")


if __name__ == "__main__":
    main()
