"""
Tầng 3 (phần 2) — lấy mẫu ngẫu nhiên nhãn B0 theo 3 mức confidence để đọc tay,
in ra câu hỏi + quote đã trích + đoạn matched_span + tên văn bản, phục vụ
đánh giá thủ công độ tin cậy của nhãn ở từng mức confidence.

Chạy: python pipeline/sample_b0_audit.py
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config, io_utils

BUCKETS = [("thấp (0.3-0.5)", 0.3, 0.5), ("trung (0.5-0.7)", 0.5, 0.7), ("cao (0.7-1.0)", 0.7, 1.01)]
N_PER_BUCKET = 10
SEED = 42


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    warmup_labels = load_json(config.OUTPUTS_DIR / "b0_labels_warmup.json")
    train_raw = load_json(config.OUTPUTS_DIR / "b0_labels_train_sample.json")
    train_labels = train_raw.get("labels", train_raw)
    qa_all = {**io_utils.load_warmup(), **io_utils.load_train()}

    all_labels = {**warmup_labels, **train_labels}

    flat = []
    for qid, spans in all_labels.items():
        for i, span in enumerate(spans):
            flat.append((qid, i, span))

    rng = random.Random(SEED)
    for name, lo, hi in BUCKETS:
        bucket = [x for x in flat if lo <= x[2]["confidence"] < hi]
        sample = rng.sample(bucket, min(N_PER_BUCKET, len(bucket)))
        print(f"\n{'=' * 90}\nBUCKET {name} — {len(bucket)} case, lấy mẫu {len(sample)}\n{'=' * 90}")
        for qid, i, span in sample:
            question = qa_all.get(qid, {}).get("question", "?")
            print(f"\n--- qid={qid}[{i}] confidence={span['confidence']:.3f} context={span['context_id']} "
                  f"khoan={span['khoan_id']} role={span['role']} ---")
            print(f"CÂU HỎI: {question}")
            print(f"MATCHED_SPAN: {span['matched_span'][:400]}")


if __name__ == "__main__":
    main()
