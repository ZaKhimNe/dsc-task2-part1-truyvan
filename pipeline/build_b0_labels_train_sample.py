"""
B0 — chạy label_dataset trên MẪU NGẪU NHIÊN của data/train.json (không phải
warmup.json, đã xong ở pipeline/build_b0_labels.py). Dùng để có thêm dữ
liệu calibrate ngưỡng confidence + đo recall trần — không cần chạy hết
7,000 câu (mất ~3 giờ), 1,500 câu ngẫu nhiên đủ thống kê (~35-40 phút, xem
docs/EXPERIMENT_LOG.md entry [B0] 13/08 tốc độ đo được với Layer 1.5:
1.56s/câu).

Chạy: python pipeline/build_b0_labels_train_sample.py
"""
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config, io_utils
from src.b2_retrieval import retrieve
from src.b0_autolabel import label

DOC_INDEX_PATH = config.OUTPUTS_DIR / "bm25_doc_index.pkl"
KHOAN_INDEX_PATH = config.OUTPUTS_DIR / "bm25_khoan_index.pkl"
OUT_PATH = config.OUTPUTS_DIR / "b0_labels_train_sample.json"
SAMPLE_SIZE = 1500
SEED = 42


def main():
    print(f"Đang load BM25 index (doc-level) từ {DOC_INDEX_PATH} ...", flush=True)
    t0 = time.time()
    bm25_index = retrieve.load_index(DOC_INDEX_PATH)
    print(f"  -> {time.time() - t0:.1f}s", flush=True)

    print(f"Đang load BM25 index (Khoản-level, Layer 1.5) từ {KHOAN_INDEX_PATH} ...", flush=True)
    t0 = time.time()
    khoan_index = retrieve.load_index(KHOAN_INDEX_PATH)
    khoan_position_index = retrieve.build_khoan_position_index(khoan_index)
    print(f"  -> {time.time() - t0:.1f}s, {len(khoan_position_index)} unit", flush=True)

    print("Đang load parsed_corpus.jsonl (B1) ...", flush=True)
    t0 = time.time()
    parsed_corpus = io_utils.load_parsed_corpus()
    print(f"  -> {len(parsed_corpus)} văn bản, {time.time() - t0:.1f}s", flush=True)

    train = io_utils.load_train()
    random.seed(SEED)
    sample_ids = random.sample(list(train.keys()), min(SAMPLE_SIZE, len(train)))
    print(f"\nĐang gán nhãn B0 cho {len(sample_ids)}/{len(train)} câu train.json "
          f"(seed={SEED}, tái lập được) ...", flush=True)

    t0 = time.time()
    labels = {}
    for i, qid in enumerate(sample_ids):
        labels[qid] = label.label_answer(
            train[qid], bm25_index, parsed_corpus, khoan_index=khoan_index, khoan_position_index=khoan_position_index
        )
        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(sample_ids)}, {time.time() - t0:.1f}s", flush=True)
    print(f"Xong: {time.time() - t0:.1f}s", flush=True)

    n_labeled = sum(1 for v in labels.values() if v)
    all_scores = [m["confidence"] for spans in labels.values() for m in spans]
    print(f"\nCó nhãn: {n_labeled}/{len(sample_ids)} ({n_labeled / len(sample_ids) * 100:.1f}%)", flush=True)
    if all_scores:
        print(f"Score: min={min(all_scores):.3f} max={max(all_scores):.3f} "
              f"mean={sum(all_scores)/len(all_scores):.3f}", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"sample_ids": sample_ids, "seed": SEED, "labels": labels}, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
