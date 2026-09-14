"""
Demo B4 — select_span() đối chiếu với nhãn B0 (matched_span) trên câu hỏi
thật, để kiểm tra sơ bộ B4 có chọn đúng vùng lân cận không.

LƯU Ý: đây KHÔNG phải benchmark chính thức của B4 — B0 định vị VĂN BẢN
nguồn (document-level), B4 cần định vị ĐÚNG ĐIỂM/CÂU (finer-grained), nên
"khớp" ở đây chỉ là proxy thô (substring overlap). Số liệu chính thức cần
nhãn gold thật ở granularity Điểm/câu — chưa có, xem README.

Input: experiments/_b3_step1_output.json (đã sinh ở bước B3 step1, chứa
labels từ B0 trên 100 câu hỏi thật).

Chạy: python experiments/demo_b4_span_selection.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import io_utils
from src.b1_parser import parser
from src.b4_span_selection import selection

IN_PATH = Path(__file__).resolve().parent / "_b3_step1_output.json"


def _find_khoan_containing(dieu_list, matched_span: str) -> str | None:
    """Tìm text Khoản (hoặc toàn Điều nếu không chia Khoản) có khả năng chứa
    matched_span nhất — kiểm tra bằng cách xem đầu/cuối matched_span có nằm
    trong text Khoản không (matched_span có thể có whitespace khác do B0
    tokenize lại, nên không so khớp y hệt toàn chuỗi)."""
    for dieu in dieu_list:
        candidates = dieu["khoan"] if dieu["khoan"] else [{"khoan_so": "", "text": dieu["text"]}]
        for k in candidates:
            if matched_span[:40] in k["text"] or matched_span[-40:] in k["text"]:
                return k["text"]
    return None


def main():
    with open(IN_PATH, encoding="utf-8") as f:
        data = json.load(f)
    warmup = io_utils.load_warmup()
    labels = data["labels"]

    n_tested = 0
    n_skipped = 0
    n_hit = 0
    for qid, spans in labels.items():
        if not spans:
            continue
        span = spans[0]
        question = warmup[qid]["question"]
        passage = io_utils.load_context(span["context_id"])["passage"]
        dieu_list = parser.parse_document(str(span["context_id"]), passage)["dieu"]

        khoan_text = _find_khoan_containing(dieu_list, span["matched_span"])
        if khoan_text is None:
            n_skipped += 1
            continue

        n_tested += 1
        selected = selection.select_span(question, khoan_text)
        matched = span["matched_span"].strip()
        hit = matched[:30] in selected or selected[:30] in matched
        n_hit += hit
        print(f"{qid}: hit={hit}")
        print(f"  question : {question[:90]}")
        print(f"  B0 matched : {matched[:100]!r}")
        print(f"  B4 selected: {selected[:100]!r}")
        print()

    print(f"=== Tổng: {n_tested} test (bỏ qua {n_skipped} do granularity B1 khác biệt), "
          f"B4 khớp vùng B0={n_hit}/{n_tested} ({n_hit / n_tested * 100:.0f}%) ===")


if __name__ == "__main__":
    main()
