"""
Demo B0 — Auto-label Alignment (rebuild 12/08/2026, dùng search_units B2
thay cho shingle index tự viết cũ). Chạy: python experiments/demo_b0_autolabel.py

Mục đích: chứng minh align_span + label_answer hoạt động đúng qua 4 tình
huống kiểm tra, trên SUBSET nhỏ (10 văn bản) — không cần chờ build_index
full corpus (việc đó chạy riêng, xem experiments/build_b0_labels.py).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import io_utils
from src.b1_parser import parser
from src.b2_retrieval import retrieve
from src.b0_autolabel import label

SUBSET_IDS = [740, 100050, 100062, 100109, 100125, 100139, 100325, 100363, 100380, 100419]


def _build_subset_parsed_corpus(ids: list[int]) -> dict[str, dict]:
    """Parse trực tiếp subset (không cần data/parsed_corpus.jsonl đầy đủ) —
    demo tự chạy được độc lập, không phụ thuộc file đã build sẵn."""
    result = {}
    for cid in ids:
        doc = io_utils.load_context(cid)
        parsed = parser.parse_document(str(cid), doc["passage"])
        result[str(cid)] = {
            "context_id": str(cid),
            "name": doc["name"],
            "link": doc["link"],
            "parse_status": parsed["parse_status"],
            "dieu": parsed["dieu"],
            "phu_luc_raw": parsed["phu_luc_raw"],
        }
    return result


def main():
    print(f"Đang tải {len(SUBSET_IDS)} văn bản thật + parse (B1) + build BM25 (B2) ...")
    docs = {i: io_utils.load_context(i)["passage"] for i in SUBSET_IDS}
    bm25_index = retrieve.build_index(docs)
    parsed_corpus = _build_subset_parsed_corpus(SUBSET_IDS)
    p740 = docs[740]

    print("\n=== Test 1: đoạn trích NGUYÊN VĂN từ Khoản 2 Điều 2 context_740 (kỳ vọng score cao) ===")
    khoan_2_1_text = next(
        k["text"] for d in parsed_corpus["740"]["dieu"] if d["dieu_so"] == "2" for k in d["khoan"] if k["khoan_so"] == "1"
    )
    excerpt = khoan_2_1_text[:200]
    span, score = label.align_span(excerpt, khoan_2_1_text)
    print(f"score={score:.3f}")
    print(f"khớp đúng nguồn: {'CÓ' if score > 0.95 else 'KHÔNG — CÓ VẤN ĐỀ'}")

    print("\n=== Test 2: đoạn trích có nhiễu nhẹ (đổi 2 từ, kỳ vọng score giảm nhưng vẫn > 0.5) ===")
    noisy = excerpt.replace("Bộ Y tế", "BỘ Y TẾ", 1).replace("quản lý", "quan ly", 1)
    span2, score2 = label.align_span(noisy, khoan_2_1_text)
    print(f"score={score2:.3f}  ({'OK, vẫn nhận diện được' if 0.5 < score2 < 1.0 else 'BẤT THƯỜNG'})")

    print("\n=== Test 3: answer có marker 'Trước đây...' (kỳ vọng tách 2 role khác nhau) ===")
    fake_answer = (
        f"Theo quy định hiện hành: {excerpt}\n"
        f"Trước đây, căn cứ văn bản cũ quy định như sau: {excerpt}"
    )
    result = label.label_answer({"question": "demo", "answer": fake_answer}, bm25_index, parsed_corpus)
    for m in result:
        print(f"  role={m['role']!r:<12} context_id={m['context_id']} khoan_id={m['khoan_id']} "
              f"unit_type={m['unit_type']} confidence={m['confidence']:.3f}")
    roles = {m["role"] for m in result}
    print(f"tách đúng 2 role: {'CÓ' if roles == {'hiện hành', 'trước đây'} else 'KHÔNG — CÓ VẤN ĐỀ'}")

    print("\n=== Test 4: đối chứng âm — câu hỏi thật KHÔNG liên quan 10 văn bản test ===")
    warmup = io_utils.load_warmup()
    qid, item = next(iter(warmup.items()))
    result_neg = label.label_answer(item, bm25_index, parsed_corpus)
    print(f"question: {item['question']}")
    print(f"result: {result_neg}")
    print(f"không dương tính giả: {'CÓ' if result_neg == [] else 'KHÔNG — CÓ VẤN ĐỀ, xem lại ngưỡng confidence'}")


if __name__ == "__main__":
    main()
