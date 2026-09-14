"""
Xây index "số hiệu văn bản" -> context_id cho toàn bộ corpus — tiện ích dùng
lại được, phục vụ `build_citation_labels.py` (match citation_metadata từ
`data123/` với corpus của mình).

Cơ chế: hầu hết văn bản pháp luật có dòng "Số: <số hiệu>" ngay đầu passage
(vd "Số: 219/2013/TT-BTC"). Trích bằng regex, chuẩn hoá, build dict ngược.

QUAN TRỌNG (bài học đã dính thật khi build lần đầu): character class của
regex PHẢI gồm cả ký tự có dấu "Đ" (xuất hiện trong "NĐ-CP", "BLĐTBXH"...) —
thiếu ký tự này làm tỉ lệ match tụt từ 99,0% xuống còn 35,5% vì số hiệu bị
cắt cụt giữa chừng (vd "41/2018/NĐ-CP" bị cắt còn "41/2018/N").

Số hiệu bị TRÙNG (>1 context_id) coi là KHÔNG DÙNG ĐƯỢC (ambiguous) — không
đoán bừa chọn 1 trong nhiều văn bản trùng số hiệu.

Chạy: python pipeline/build_doc_number_index.py
"""
import json
import re
import unicodedata
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import config

CORPUS_DIR = config.DATA_DIR / "corpus"
OUT_PATH = config.OUTPUTS_DIR / "doc_number_index.json"

_SO_RE = re.compile(r"S[ốôo]\s*:?[\s\r\n]*([0-9]+[A-Za-z0-9\-/ĐđƯưƠơ]*)", re.IGNORECASE)


def normalize_num(s: str) -> str:
    s = re.sub(r"\s+", "", s)
    return unicodedata.normalize("NFC", s).upper()


def extract_doc_number(passage: str) -> str | None:
    head = (passage or "")[:400]
    m = _SO_RE.search(head)
    if not m:
        return None
    raw = m.group(1).split("\n\n")[0].strip().rstrip(".,;")
    if not raw:
        return None
    return normalize_num(raw)


def main():
    print("Đang quét toàn bộ corpus...", flush=True)
    raw_index: dict[str, list[str]] = {}
    n_total = 0
    n_extracted = 0

    for fp in sorted(CORPUS_DIR.glob("context_*.json")):
        n_total += 1
        with open(fp, encoding="utf-8") as f:
            d = json.load(f)
        num = extract_doc_number(d.get("passage"))
        if num:
            n_extracted += 1
            raw_index.setdefault(num, []).append(str(d.get("id")))

    n_dup = sum(1 for v in raw_index.values() if len(v) > 1)
    usable_index = {k: v[0] for k, v in raw_index.items() if len(v) == 1}
    ambiguous = {k: v for k, v in raw_index.items() if len(v) > 1}

    print(f"Tổng văn bản: {n_total}", flush=True)
    print(f"Trích được số hiệu: {n_extracted}/{n_total} = {n_extracted / n_total:.1%}", flush=True)
    print(f"Số hiệu duy nhất (dùng được): {len(usable_index)}", flush=True)
    print(f"Số hiệu bị trùng (ambiguous, KHÔNG dùng): {n_dup}", flush=True)
    print("\nVí dụ 5 số hiệu dùng được:", flush=True)
    for k in list(usable_index.keys())[:5]:
        print(f"  {k} -> context_id {usable_index[k]}", flush=True)
    print("\nVí dụ 5 số hiệu bị trùng:", flush=True)
    for k in list(ambiguous.keys())[:5]:
        print(f"  {k} -> context_id {ambiguous[k]}", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "usable": usable_index,
            "ambiguous": ambiguous,
            "n_total_corpus": n_total,
            "n_extracted": n_extracted,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
