"""
B0 — chạy label_dataset trên FULL warmup.json (500 câu), dùng BM25 index đã
cache (pipeline/build_bm25_index.py) + bm25_khoan_index.pkl (Layer 1.5,
pipeline/build_expensive_indexes.py L1_khoan) + parsed_corpus.jsonl (B1).
Ghi kết quả ra outputs/b0_labels_warmup.json.

REBUILD 13/08/2026: thêm Layer 1.5 (BM25 rerank cấp Khoản, xem
docs/EXPERIMENT_LOG.md entry [B2] 13/08) vào candidate retrieval của B0 —
đo được "được cả 2" trên data_retrieve (độ sắc nét gần bằng index Khoản
thẳng, giữ nguyên coverage Layer 1) nên kỳ vọng B0 tìm nhãn tốt hơn/nhiều
hơn bản trước (61.6% coverage, dùng candidate KHÔNG rerank).

Chạy: python pipeline/build_b0_labels.py (cần đã có bm25_doc_index.pkl và
bm25_khoan_index.pkl — chạy build_bm25_index.py + build_expensive_indexes.py
L1_khoan trước nếu chưa có).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config, io_utils
from src.b2_retrieval import retrieve
from src.b0_autolabel import label

DOC_INDEX_PATH = config.OUTPUTS_DIR / "bm25_doc_index.pkl"
KHOAN_INDEX_PATH = config.OUTPUTS_DIR / "bm25_khoan_index.pkl"
OUT_PATH = config.OUTPUTS_DIR / "b0_labels_warmup.json"


def main():
    print(f"Đang load BM25 index (doc-level) từ {DOC_INDEX_PATH} ...", flush=True)
    t0 = time.time()
    bm25_index = retrieve.load_index(DOC_INDEX_PATH)
    print(f"  -> {time.time() - t0:.1f}s", flush=True)

    khoan_index = None
    khoan_position_index = None
    if KHOAN_INDEX_PATH.exists():
        print(f"Đang load BM25 index (Khoản-level, Layer 1.5) từ {KHOAN_INDEX_PATH} ...", flush=True)
        t0 = time.time()
        khoan_index = retrieve.load_index(KHOAN_INDEX_PATH)
        khoan_position_index = retrieve.build_khoan_position_index(khoan_index)
        print(f"  -> {time.time() - t0:.1f}s, {len(khoan_position_index)} unit", flush=True)
    else:
        print("KHÔNG có bm25_khoan_index.pkl -> B0 chạy KHÔNG có Layer 1.5 (candidate như bản cũ).", flush=True)

    print("Đang load parsed_corpus.jsonl (B1) ...", flush=True)
    t0 = time.time()
    parsed_corpus = io_utils.load_parsed_corpus()
    print(f"  -> {len(parsed_corpus)} văn bản, {time.time() - t0:.1f}s", flush=True)

    warmup = io_utils.load_warmup()
    print(f"\nĐang gán nhãn B0 cho full {len(warmup)} câu hỏi warmup.json ...", flush=True)
    t0 = time.time()
    labels = {}
    for i, (qid, item) in enumerate(warmup.items()):
        labels[qid] = label.label_answer(
            item, bm25_index, parsed_corpus, khoan_index=khoan_index, khoan_position_index=khoan_position_index
        )
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(warmup)}, {time.time() - t0:.1f}s", flush=True)
    print(f"Xong: {time.time() - t0:.1f}s", flush=True)

    n_labeled = sum(1 for v in labels.values() if v)
    all_scores = [m["confidence"] for spans in labels.values() for m in spans]
    print(f"\nCó nhãn: {n_labeled}/{len(warmup)} ({n_labeled / len(warmup) * 100:.1f}%)", flush=True)
    if all_scores:
        print(f"Score: min={min(all_scores):.3f} max={max(all_scores):.3f} "
              f"mean={sum(all_scores)/len(all_scores):.3f}", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
