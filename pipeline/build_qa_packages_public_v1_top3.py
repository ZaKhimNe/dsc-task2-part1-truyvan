"""
Build QA packages v1 (cấu hình cũ: top-3, chỉ Khoản) cho 1.000 câu public-official,
để so sánh với v2 (top-1, mở rộng Điều).

V1 = Baseline trước khi thay đổi TOP_N (31/08). Giữ nguyên TOP_N=3 và chỉ trả text
Khoản gốc (không expand Điều).

Chạy: python pipeline/build_qa_packages_public_v1_top3.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config, io_utils
from src.b2_retrieval.retrieve import get_unit_text
from src.b4_span_selection.selection import select_top_n_khoan
from src.b6_context_package.package import build_context_item, build_qa_package, write_qa_packages_json

RERANK_PATH = Path("outputs/layer3/L3_rerank_public.jsonl")
OUT_PATH_JSON = config.OUTPUTS_DIR / "qa_packages_public_v1_top3.json"
TOP_N = 3


def infer_unit_type(unit_id: str, context_id: str) -> str:
    suffix = unit_id[len(context_id) + 1 :] if unit_id.startswith(context_id + "_") else ""
    n_parts = len([p for p in suffix.split("_") if p])
    if n_parts >= 2:
        return "khoan"
    if n_parts == 1:
        return "dieu_fallback"
    return "doc_fallback"


def main():
    print("Dang load parsed_corpus + public-official.json + ket qua Layer 3...", flush=True)
    parsed_corpus = io_utils.load_parsed_corpus()
    qa_public = io_utils.load_public_official()

    with open(config.OUTPUTS_DIR / "doc_number_index.json", encoding="utf-8") as f:
        doc_number_index = json.load(f)["usable"]
    context_id_to_doc_number = {v: k for k, v in doc_number_index.items()}

    n_empty_text = 0
    packages = []
    with open(RERANK_PATH, encoding="utf-8") as f_in:
        for line in f_in:
            r = json.loads(line)
            qid = r["qid"]
            question = qa_public[qid]["question"]
            reference_answer = qa_public[qid].get("answer")

            khoan_list = []
            for u in r["ranked_units"]:
                text = get_unit_text(parsed_corpus, u["unit_id"])
                khoan_list.append({
                    "unit_id": u["unit_id"],
                    "context_id": u["context_id"],
                    "text": text,
                    "score": u["score"],
                    "unit_type": infer_unit_type(u["unit_id"], u["context_id"]),
                })

            top_units = select_top_n_khoan(question, khoan_list, top_n=TOP_N, already_ranked=True)

            context_items = []
            for u in top_units:
                doc = parsed_corpus.get(u["context_id"], {})
                # V1: NO context expansion — just use original Khoan text
                context_text = u["text"]
                context_items.append(build_context_item(
                    context_id=u["context_id"], unit_id=u["unit_id"], unit_type=u["unit_type"],
                    text=context_text, source_name=doc.get("name", ""), source_link=doc.get("link", ""),
                    document_number=context_id_to_doc_number.get(u["context_id"], ""),
                    retrieval_score=u["score"],
                ))
                if not context_text.strip():
                    n_empty_text += 1

            pkg = build_qa_package(qid, question, context_items, reference_answer)
            packages.append(pkg)

    n_written = write_qa_packages_json(packages, OUT_PATH_JSON)
    print(f"Da ghi {n_written} QAPackage -> {OUT_PATH_JSON}", flush=True)
    print(f"Kiem tra: so context co text RONG = {n_empty_text} (ky vong 0)", flush=True)
    print(f"Kich thuoc file: {OUT_PATH_JSON.stat().st_size / 1024:.1f} KB", flush=True)


if __name__ == "__main__":
    main()
