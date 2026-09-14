"""
Tính bù embedding cho các unit mà Layer 2 (Kaggle) đã tính TRÊN CHUỖI RỖNG.

Bối cảnh (xem docs/EXPERIMENT_LOG.md entry [B1] 17/08): B1 có bug khiến
7,163 unit (1.66%) có `text` rỗng — nội dung bị hút hết vào `dieu_tieu_de`
với các Điều ngắn nằm cùng dòng header. `embed_corpus_notebook.py` dùng
`dieu["text"]` nên embedding của đúng 7,163 unit đó là vector RÁC (embedding
của chuỗi rỗng). B1 đã sửa + rebuild `parsed_corpus.jsonl`, giờ cần tính lại
embedding CHỈ cho các unit đó rồi ghi đè đúng các hàng trong
`outputs/layer2/corpus_embeddings.npy` — rẻ hơn nhiều so với chạy lại toàn bộ
432k unit trên Kaggle.

Cách xác định unit cần vá: so text HIỆN TẠI (sau khi B1 sửa) với thứ tự
`corpus_unit_ids.json` — unit nào giờ có text mà lúc Kaggle chạy là rỗng thì
cần vá. Vì không lưu lại text cũ, dùng dấu hiệu chắc chắn tương đương: unit
là Dieu KHÔNG chia Khoản và `text == dieu_tieu_de` (đúng dạng đã sửa).

Chạy: python pipeline/patch_layer2_embeddings.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config, io_utils

LAYER2_DIR = config.OUTPUTS_DIR / "layer2"
EMB_PATH = LAYER2_DIR / "corpus_embeddings.npy"
BACKUP_PATH = LAYER2_DIR / "corpus_embeddings.before_patch.npy"
MODEL_NAME = "BAAI/bge-m3"


def main():
    print("Đang load parsed_corpus + corpus_unit_ids...", flush=True)
    parsed_corpus = io_utils.load_parsed_corpus()
    with open(LAYER2_DIR / "corpus_unit_ids.json", encoding="utf-8") as f:
        corpus_unit_ids = json.load(f)

    # Xây map unit_id -> text theo ĐÚNG cách embed_corpus_notebook.py CELL 4 sinh ra,
    # kèm cờ "unit này thuộc dạng đã bị sửa" (Dieu không chia Khoản, text == tiêu đề)
    need_patch_ids: list[str] = []
    text_by_id: dict[str, str] = {}
    for cid, doc in parsed_corpus.items():
        if doc["parse_status"] == "fallback":
            if doc["dieu"]:
                d0 = doc["dieu"][0]
                text_by_id[d0["dieu_id"] or cid] = d0["text"]
            continue
        for dieu in doc["dieu"]:
            if dieu["khoan"]:
                for k in dieu["khoan"]:
                    text_by_id.setdefault(k["khoan_id"], k["text"])
            else:
                text_by_id.setdefault(dieu["dieu_id"], dieu["text"])
                if dieu["text"] and dieu["text"] == dieu["dieu_tieu_de"]:
                    need_patch_ids.append(dieu["dieu_id"])

    need_patch_ids = sorted(set(need_patch_ids))
    print(f"  -> {len(corpus_unit_ids)} unit trong embedding, "
          f"{len(need_patch_ids)} unit cần vá", flush=True)
    if not need_patch_ids:
        print("Không có gì cần vá — dừng.")
        return

    id_to_row = {uid: i for i, uid in enumerate(corpus_unit_ids)}
    rows: list[int] = []
    texts: list[str] = []
    for uid in need_patch_ids:
        row = id_to_row.get(uid)
        if row is None:
            continue  # unit mới xuất hiện sau khi Kaggle chạy — không có hàng để vá
        rows.append(row)
        texts.append(text_by_id[uid])
    print(f"  -> vá {len(rows)} hàng trong ma trận embedding", flush=True)

    print(f"Đang load model {MODEL_NAME} (CPU, lần đầu tải ~1.2GB)...", flush=True)
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.max_seq_length = 1024
    print(f"  -> {time.time() - t0:.1f}s", flush=True)

    print(f"Đang embed {len(texts)} text (text ngắn — tiêu đề Điều)...", flush=True)
    t0 = time.time()
    new_emb = model.encode(texts, batch_size=32, show_progress_bar=True,
                            normalize_embeddings=True, convert_to_numpy=True)
    print(f"  -> {time.time() - t0:.1f}s", flush=True)

    print(f"Đang load {EMB_PATH} ...", flush=True)
    emb = np.load(EMB_PATH)
    print(f"  shape={emb.shape} dtype={emb.dtype}", flush=True)

    if not BACKUP_PATH.exists():
        print(f"Sao lưu bản gốc -> {BACKUP_PATH}", flush=True)
        np.save(BACKUP_PATH, emb)
    else:
        print(f"(đã có sao lưu {BACKUP_PATH.name}, không ghi lại)", flush=True)

    emb[np.array(rows)] = new_emb.astype(emb.dtype)
    np.save(EMB_PATH, emb)
    print(f"Đã ghi đè {len(rows)} hàng -> {EMB_PATH}", flush=True)

    # sanity: các hàng vừa vá phải có norm ~1 (đã normalize) và khác nhau
    patched = emb[np.array(rows[: min(200, len(rows))])].astype(np.float32)
    norms = np.linalg.norm(patched, axis=1)
    n_unique = len({tuple(np.round(v[:8], 4)) for v in patched})
    print(f"Sanity: norm trung bình={norms.mean():.4f} (kỳ vọng ~1.0), "
          f"{n_unique}/{len(patched)} hàng đầu KHÁC NHAU (kỳ vọng gần bằng nhau)", flush=True)


if __name__ == "__main__":
    main()
