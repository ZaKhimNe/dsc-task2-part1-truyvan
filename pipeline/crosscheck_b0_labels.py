"""
Tầng 3 (docs/BAO_CAO_TONG_HOP_B.md mục 6) — Kiểm chứng chéo độ tin của nhãn B0
bằng các câu hỏi TRÙNG NỘI DUNG giữa 2 track: `data_retrieve` (nhãn văn bản
CHO SẴN, sạch, không qua B2) và `data/train.json`+`warmup.json` (câu hỏi QA
đang làm, phải suy nhãn qua B0). Nếu B0 dự đoán đúng context_id mà
`data_retrieve` xác nhận sẵn, đó là bằng chứng độc lập cho độ tin của B0
(không vòng tròn — data_retrieve không đi qua B2/B0).

Cách tìm case trùng: khớp NGUYÊN VĂN câu hỏi (`.strip()`) giữa 2 bộ — đo
được 29 cặp (một câu QA có thể trùng >1 câu data_retrieve nếu data_retrieve
hỏi lại nhiều lần, hoặc ngược lại).

Với mỗi cặp: lấy gold context_id (union nếu >1 câu data_retrieve trùng),
lấy nhãn B0 cho câu QA (ưu tiên đọc từ output đã có sẵn
`outputs/b0_labels_warmup.json`/`outputs/b0_labels_train_sample.json`; nếu
câu đó chưa từng được B0 gán nhãn thì chạy `label_answer` sống ngay tại đây
— chỉ có tối đa 29 câu nên rẻ).

Chạy: python pipeline/crosscheck_b0_labels.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config, io_utils
from src.b2_retrieval.retrieve import load_index, build_khoan_position_index
from src.b0_autolabel.label import label_answer


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_overlap():
    dr_all = {
        **load_json(config.DATA_RETRIEVE_DIR / "train.json"),
        **load_json(config.DATA_RETRIEVE_DIR / "warmup.json"),
    }
    qa_all = {**io_utils.load_train(), **io_utils.load_warmup()}

    dr_by_q: dict[str, list[str]] = {}
    for qid, item in dr_all.items():
        dr_by_q.setdefault(item["question"].strip(), []).append(qid)

    overlap = []
    for qid, item in qa_all.items():
        q = item["question"].strip()
        if q in dr_by_q:
            gold = set()
            for dr_qid in dr_by_q[q]:
                gold.update(str(c) for c in dr_all[dr_qid]["answer"])
            overlap.append((qid, item, gold))
    return overlap


def load_existing_b0_labels():
    labels: dict[str, list] = {}
    warmup_raw = load_json(config.OUTPUTS_DIR / "b0_labels_warmup.json")
    labels.update(warmup_raw)
    train_raw = load_json(config.OUTPUTS_DIR / "b0_labels_train_sample.json")
    labels.update(train_raw.get("labels", train_raw))
    return labels


def main():
    overlap = find_overlap()
    print(f"Số cặp câu hỏi trùng nội dung (QA <-> data_retrieve): {len(overlap)}", flush=True)

    existing_labels = load_existing_b0_labels()
    missing = [(qid, item) for qid, item, _ in overlap if qid not in existing_labels]
    print(f"Đã có nhãn B0 sẵn: {len(overlap) - len(missing)}, cần chạy sống: {len(missing)}", flush=True)

    live_labels: dict[str, list] = {}
    if missing:
        print("Đang load BM25 index + parsed_corpus để chạy B0 sống...", flush=True)
        bm25_index = load_index(config.OUTPUTS_DIR / "bm25_doc_index.pkl")
        khoan_index = load_index(config.OUTPUTS_DIR / "bm25_khoan_index.pkl")
        khoan_position_index = build_khoan_position_index(khoan_index)
        parsed_corpus = io_utils.load_parsed_corpus()
        for qid, item in missing:
            live_labels[qid] = label_answer(
                item, bm25_index, parsed_corpus, khoan_index=khoan_index,
                khoan_position_index=khoan_position_index,
            )

    n_total = 0
    n_hit = 0
    n_no_label = 0
    rows = []
    for qid, item, gold in overlap:
        spans = existing_labels.get(qid, live_labels.get(qid, []))
        if not spans:
            n_no_label += 1
            continue
        n_total += 1
        pred_context_ids = {s["context_id"] for s in spans}
        hit = bool(pred_context_ids & gold)
        n_hit += hit
        rows.append((qid, item["question"][:70], gold, pred_context_ids, hit))

    print(f"\n=== Kết quả (n={n_total} câu có nhãn B0 để so, {n_no_label} câu B0 không gán được nhãn) ===")
    print(f"B0 dự đoán TRÚNG context_id mà data_retrieve xác nhận: {n_hit}/{n_total} = "
          f"{n_hit / n_total:.1%}" if n_total else "n/a")
    print()
    for qid, q, gold, pred, hit in rows:
        mark = "OK" if hit else "SAI"
        print(f"[{mark}] qid={qid} gold={sorted(gold)} B0_pred={sorted(pred)} | {q}")


if __name__ == "__main__":
    main()
