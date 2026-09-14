"""
Demo B2 — Layer 1 (BM25 cấp văn bản) + Expand (Khoản/Dieu). Chạy:
python experiments/demo_b2_retrieval.py

Test 1: build_query_variants — kiểm tra sinh biến thể đúng.
Test 2: search_docs_multi tự tìm lại context_740 (câu lấy thẳng từ nội dung).
Test 3: expand_to_units trên context_740 — đúng 5 Điều, unit_type="khoan"
  cho Điều có Khoản, "dieu_fallback" cho Điều không chia Khoản (Điều 1, 5).
Test 4: search_units gộp cả 2 bước, chạy trên câu hỏi thật từ warmup.json.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import io_utils
from src.b2_retrieval import retrieve

SUBSET_IDS = [740, 100050, 100062, 100109, 100125, 100139, 100325, 100363, 100380, 100419]


def test_query_variants():
    print("=== Test 1: build_query_variants ===")
    v1 = retrieve.build_query_variants("Vụ Trang thiết bị làm gì?")
    assert "Vụ Trang thiết bị làm gì?" in v1
    v2 = retrieve.build_query_variants(
        "Câu đầu ngắn. Đây là câu dài nhất chứa nhiều thông tin hơn hẳn các câu khác trong đoạn. Câu cuối."
    )
    assert len(v2) == 4, f"kỳ vọng 4 biến thể khác nhau, được {len(v2)}: {v2}"
    print(f"  1 câu -> {len(v1)} biến thể (trùng, đã dedup); nhiều câu -> {len(v2)} biến thể")
    print("  PASS\n")


def test_search_docs_multi():
    print("=== Test 2: search_docs_multi tự tìm lại context_740 ===")
    docs = {i: io_utils.load_context(i)["passage"] for i in SUBSET_IDS}
    index = retrieve.build_index(docs)
    query = "Vụ Trang thiết bị và Công trình y tế thuộc Bộ Y tế"
    results = retrieve.search_docs_multi(index, query, top_k_docs=5)
    top1 = results[0]["context_id"]
    print(f"  top1={top1} {'(ĐÚNG)' if top1 == 740 else '(SAI, kỳ vọng 740)'}")
    assert top1 == 740
    print("  PASS\n")
    return index


def test_expand_to_units(index):
    print("=== Test 3: expand_to_units trên context_740 ===")
    parsed_corpus = io_utils.load_parsed_corpus()
    doc_results = [{"context_id": 740, "score": 10.0}]
    units = retrieve.expand_to_units(doc_results, parsed_corpus)
    print(f"  {len(units)} unit từ context_740")
    for u in units:
        print(f"    {u['unit_id']:>10}  type={u['unit_type']:<14} len={len(u['text'])}")
    khoan_units = [u for u in units if u["unit_type"] == "khoan"]
    fallback_units = [u for u in units if u["unit_type"] == "dieu_fallback"]
    assert len(khoan_units) == 8, f"kỳ vọng 8 Khoản (Điều 2,3,4), được {len(khoan_units)}"
    assert len(fallback_units) == 2, f"kỳ vọng 2 dieu_fallback (Điều 1, 5), được {len(fallback_units)}"
    assert all(u["context_id"] == "740" for u in units)
    print("  PASS\n")
    return parsed_corpus


def test_search_units(index, parsed_corpus):
    print("=== Test 4: search_units trên câu hỏi thật từ warmup.json ===")
    warmup = io_utils.load_warmup()
    qid, item = next(iter(warmup.items()))
    print(f"  Câu hỏi: {item['question'][:100]}")
    t0 = time.time()
    units = retrieve.search_units(index, item["question"], parsed_corpus, top_k_docs=5)
    print(f"  -> {len(units)} unit ứng viên trong {time.time() - t0:.2f}s "
          f"(không có nhãn B0 trong subset nhỏ này nên không kỳ vọng khớp)")
    assert isinstance(units, list)
    print("  PASS\n")


if __name__ == "__main__":
    test_query_variants()
    index = test_search_docs_multi()
    parsed_corpus = test_expand_to_units(index)
    test_search_units(index, parsed_corpus)
    print("Tất cả test PASS.")
