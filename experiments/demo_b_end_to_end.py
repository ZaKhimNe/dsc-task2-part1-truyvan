"""
Demo end-to-end theo THIẾT KẾ CUỐI của B2 (chốt 17/08/2026 — mọi lựa chọn có
số đo hậu thuẫn, xem docs/EXPERIMENT_LOG.md entry [B2/B4] 17/08):

  search_units_hybrid()  = Layer 1 (BM25 doc) ∥ Layer 2 (dense unit-level)
                            → Union cấp unit → xếp hạng bằng dense → top-50
  → B4 select_top_n_khoan (top-3, đã sort sẵn nên already_ranked=True)
  → B6 build_qa_package (schema Generator)

KHÁC bản trước (13/08) ở 3 điểm, mỗi điểm có số đo:
  1. Dùng `search_units_hybrid` (union 2 nhánh) thay vì chỉ Layer 1 + rerank.
  2. KHÔNG ràng buộc đa dạng (`max_per_doc`) — đo được `=1` làm Hit Khoản sụp
     81.9% → 43.3% (giữ đúng văn bản nhưng SAI Khoản).
  3. KHÔNG gọi `select_span` — đo được nó XOÁ MẤT đoạn đáp án 55.6% số lần;
     trả NGUYÊN Khoản (đúng nguyên tắc "thà dư hơn thiếu" + METEOR phạt viết
     thiếu nặng gấp 3-4 lần viết dư).

Chạy: python experiments/demo_b_end_to_end.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config, io_utils
from src.b2_retrieval import retrieve
from src.b4_span_selection import selection
from src.b6_context_package import package

LAYER2_DIR = config.OUTPUTS_DIR / "layer2"
N_QUESTIONS_DEMO = 5
TOP_N_CONTEXTS = 3


def main():
    print("Đang load BM25 index + parsed_corpus + embedding Layer 2 ...")
    t0 = time.time()
    bm25_index = retrieve.load_index(config.OUTPUTS_DIR / "bm25_doc_index.pkl")
    parsed_corpus = io_utils.load_parsed_corpus()
    corpus_emb = np.load(LAYER2_DIR / "corpus_embeddings.npy").astype(np.float32)
    with open(LAYER2_DIR / "corpus_unit_ids.json", encoding="utf-8") as f:
        corpus_unit_ids = json.load(f)
    emb_index = retrieve.build_corpus_embedding_index(corpus_unit_ids)
    # embedding câu hỏi đã tính sẵn trên Kaggle cho qa_train (dùng train, KHÔNG dùng
    # warmup — quyết định người dùng 17/08: đã qua giai đoạn warm-up)
    q_emb = np.load(LAYER2_DIR / "query_embeddings_qa_train.npy").astype(np.float32)
    with open(LAYER2_DIR / "query_qids_qa_train.json", encoding="utf-8") as f:
        q_qids = json.load(f)
    qid_to_qemb = dict(zip(q_qids, q_emb))
    print(f"  -> {time.time() - t0:.1f}s")

    with open(config.OUTPUTS_DIR / "doc_number_index.json", encoding="utf-8") as f:
        doc_number_index = json.load(f)["usable"]
    context_id_to_doc_number = {v: k for k, v in doc_number_index.items()}

    train = io_utils.load_train()
    demo_qids = q_qids[:N_QUESTIONS_DEMO]

    print(f"\n=== Chạy end-to-end cho {len(demo_qids)} câu hỏi thật (qa_train) ===\n")
    packages = []
    for qid in demo_qids:
        question = train[qid]["question"]
        reference_answer = train[qid].get("answer")
        print(f"[{qid}] {question[:95]}")

        t0 = time.time()
        units = retrieve.search_units_hybrid(
            bm25_index, question, parsed_corpus,
            corpus_emb, corpus_unit_ids, emb_index, qid_to_qemb[qid],
        )
        print(f"  B2(search_units_hybrid) -> {len(units)} unit, {time.time() - t0:.2f}s")

        top_units = selection.select_top_n_khoan(question, units, top_n=TOP_N_CONTEXTS,
                                                 already_ranked=True)
        context_items = []
        for u in top_units:
            doc = parsed_corpus[u["context_id"]]
            context_items.append(
                package.build_context_item(
                    context_id=u["context_id"], unit_id=u["unit_id"], unit_type=u["unit_type"],
                    text=u["text"],  # NGUYÊN Khoản — KHÔNG select_span (xem docstring đầu file)
                    source_name=doc["name"], source_link=doc["link"],
                    document_number=context_id_to_doc_number.get(u["context_id"], ""),
                    retrieval_score=u["score"],
                )
            )

        pkg = package.build_qa_package(qid, question, context_items, reference_answer)
        packages.append(pkg)
        for c in pkg["contexts"]:
            print(f"    - document_number={c['document_number']!r} article={c['article']!r} clause={c['clause']!r} "
                  f"score={c['retrieval_score']:.4f} len(text)={len(c['text'])} text={c['text'][:60]!r}")
        print()

    print("Pipeline B2(hybrid) -> B4 -> B6 chạy hết, không lỗi.")
    n_empty = sum(1 for p in packages for c in p["contexts"] if not c["text"].strip())
    print(f"Kiểm tra: số context có text RỖNG = {n_empty} (kỳ vọng 0)")
    print("\nMẫu 1 QAPackage (json, cắt 900 ký tự):")
    print(json.dumps(packages[0], ensure_ascii=False, indent=2)[:900])


if __name__ == "__main__":
    main()
