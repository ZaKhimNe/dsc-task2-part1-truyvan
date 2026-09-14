"""
Build BM25 index (B2 Layer 1) trên FULL corpus 1 lần, cache ra
outputs/bm25_doc_index.pkl — dùng lại cho B0 (build_b0_labels.py) và mọi
script khác cần index này, tránh build lại mỗi lần (~43 phút/lần đo được
12/08/2026, xem docs/EXPERIMENT_LOG.md).

Chạy: python experiments/build_bm25_index.py (nên chạy nền, mất ~43 phút)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config, io_utils
from src.b2_retrieval import retrieve

OUT_PATH = config.OUTPUTS_DIR / "bm25_doc_index.pkl"


def main():
    print("Đang tải full corpus...", flush=True)
    t0 = time.time()
    docs = {d["id"]: d["passage"] for d in io_utils.iter_corpus()}
    print(f"  -> {len(docs)} văn bản, {time.time() - t0:.1f}s", flush=True)

    print("Đang build BM25 index (rank_bm25 + underthesea) — dự kiến ~40-45 phút...", flush=True)
    t0 = time.time()
    index = retrieve.build_index(docs)
    print(f"  -> {time.time() - t0:.1f}s", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    retrieve.save_index(index, OUT_PATH)
    print(f"Đã lưu {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
