"""
B0 — gán nhãn cho 1 TẬP GIỮ KÍN THẬT SỰ (800 câu, cố định bằng seed) trích từ
`data/train.json`, lấy từ phần CHƯA từng được dùng để ra bất kỳ quyết định
thiết kế nào (loại trừ 1.500 câu dev-sample seed=42 đã "nhiễm" vì dùng lặp
lại để chốt max_per_doc, select_span, RRF...).

Mục đích: có nguồn nhãn cấp Khoản KHÔNG THIÊN VỊ để đánh giá Layer 3 ở đúng
cấp quyết định điểm cuối (xem plan resilient-snacking-firefly.md, Phase A).

QUY TẮC: outputs/b0_labels_train_heldout.json từ nay TUYỆT ĐỐI KHÔNG dùng để
tune bất cứ gì — chỉ dùng để ĐÁNH GIÁ (Phase C của plan).

CHẠY THEO LÔ (resumable): mỗi lần chạy chỉ xử lý tối đa BATCH_SIZE=200 câu
CÒN THIẾU trong 800 câu cố định, rồi dừng — gọi lại nhiều lần cho tới khi đủ
800. Tập 800 câu là CỐ ĐỊNH (seed=123, không đổi giữa các lần gọi) nên gọi
lại bao nhiêu lần cũng ra cùng 1 tập, chỉ khác là xử lý tiếp phần còn thiếu.

Tốc độ đo được trước (1.56s/câu, có Layer 1.5): 200 câu ≈ 5,2 phút/lô.

Chạy: python pipeline/build_b0_labels_train_heldout.py
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
SAMPLE_PATH = config.OUTPUTS_DIR / "b0_labels_train_sample.json"
OUT_PATH = config.OUTPUTS_DIR / "b0_labels_train_heldout.json"

HELDOUT_SIZE = 800
BATCH_SIZE = 200
SEED = 123  # khác seed=42 của dev-sample cũ, tránh trùng


def compute_heldout_ids(train: dict, already_used: set[str]) -> list[str]:
    """Tập 800 câu CỐ ĐỊNH — luôn ra cùng kết quả mỗi lần gọi (seed cố định),
    không phụ thuộc trạng thái đã label bao nhiêu."""
    pool = sorted(qid for qid in train if qid not in already_used)
    rng = random.Random(SEED)
    return rng.sample(pool, min(HELDOUT_SIZE, len(pool)))


def main():
    train = io_utils.load_train()
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        already_used = set(json.load(f)["sample_ids"])

    heldout_ids = compute_heldout_ids(train, already_used)

    existing_labels: dict = {}
    if OUT_PATH.exists():
        with open(OUT_PATH, encoding="utf-8") as f:
            existing_labels = json.load(f)["labels"]

    remaining = [qid for qid in heldout_ids if qid not in existing_labels]
    print(f"Tập giữ kín cố định: {len(heldout_ids)} câu | đã gán nhãn trước đó: "
          f"{len(heldout_ids) - len(remaining)} | còn thiếu: {len(remaining)}", flush=True)

    if not remaining:
        print("Đã xong toàn bộ tập giữ kín — không còn gì để chạy.", flush=True)
        return

    batch = remaining[:BATCH_SIZE]
    print(f"Đang load BM25 index (doc + Khoản) + parsed_corpus...", flush=True)
    t0 = time.time()
    bm25_index = retrieve.load_index(DOC_INDEX_PATH)
    khoan_index = retrieve.load_index(KHOAN_INDEX_PATH)
    khoan_position_index = retrieve.build_khoan_position_index(khoan_index)
    parsed_corpus = io_utils.load_parsed_corpus()
    print(f"  -> {time.time() - t0:.1f}s", flush=True)

    print(f"Đang gán nhãn cho lô {len(batch)} câu (tổng đã xong sẽ là "
          f"{len(heldout_ids) - len(remaining) + len(batch)}/{len(heldout_ids)})...", flush=True)
    t0 = time.time()
    for i, qid in enumerate(batch):
        existing_labels[qid] = label.label_answer(
            train[qid], bm25_index, parsed_corpus,
            khoan_index=khoan_index, khoan_position_index=khoan_position_index,
        )
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(batch) - i - 1)
            print(f"  ... {i + 1}/{len(batch)} của lô này, {elapsed:.1f}s, ETA lô {eta:.0f}s", flush=True)
    print(f"Xong lô: {time.time() - t0:.1f}s", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"heldout_ids": heldout_ids, "excluded_tuning_ids": sorted(already_used),
                    "labels": existing_labels}, f, ensure_ascii=False, indent=2)

    n_done = len(heldout_ids) - len(remaining) + len(batch)
    print(f"\nĐã lưu {OUT_PATH} — {n_done}/{len(heldout_ids)} câu đã có nhãn.", flush=True)
    if n_done < len(heldout_ids):
        print(f"CÒN {len(heldout_ids) - n_done} câu — chạy lại script này để tiếp tục.", flush=True)
    else:
        print("HOÀN TẤT toàn bộ tập giữ kín 800 câu.", flush=True)


if __name__ == "__main__":
    main()
