"""
Build QA packages v5b — GIỐNG HỆT selector của V5 (top-3, mở rộng Điều CÓ ngân
sách + dedup chuỗi y hệt), nhưng dùng RANKING MỚI (bi-encoder + reranker
AITeamVN, `L3_rerank_public_aiteamvn.jsonl`) thay vì ranking CŨ.

Đây là "V5b" trong plan — cô lập ĐÚNG 1 biến so với V5 (đổi ranking, giữ nguyên
cách đóng gói) để biết retrieval tốt hơn (đã xác nhận trên heldout: reranker
+9,1% Hit@3 ban đầu/+5,2% sau kiểm chứng lại với bộ nhãn gộp; bi-encoder +4,0%
recall@50 ban đầu/+4,5% sau kiểm chứng) có thật sự đẩy điểm LB ở Task 2 hay
không — câu hỏi chiến lược quyết định cả nhóm Phase 3/5/6 có đáng làm tiếp.

Build QA packages v5 gốc (mở rộng Điều CÓ ngân sách + dedup) cho 1.000 câu
public-official.

V5 = phiên bản sửa của "V4" (top-3 + expand toàn bộ Điều, chưa từng nộp).
V4 nguyên bản bị bỏ trước khi chạy vì 2 lý do đo được, không phải đoán:

1. Độ dài mất kiểm soát: expand thẳng top-3 Khoản ra full Điều cho ra ~7.015
   ký tự / câu (~2.000 âm tiết ước lượng) — gần gấp đôi LEN_CAP=1200 mà
   `eval_harness/baseline_template.py` đã dò thật trên dev_fast (plateau
   400-600 âm tiết, dưới LEN_FLOOR=250 thì sập điểm — đúng như V3 đo được
   734 ký tự -> 0,3535 trên LB). Không có lý do để tin 2.000 âm tiết còn nằm
   trong vùng có lợi.
2. Trùng lặp: `answers.json` (V1 thật, top-3 raw) đã cho thấy 2/3 context bị
   trùng nội dung (vd id=118909, id=160177) — generator (`build_context_bundle`,
   xem qa_generator/legalQA_Task2/src/model_production/pipeline.py) nối text
   verbatim, KHÔNG dedup. Mở rộng Điều trên top-3 làm nặng thêm lỗi này: 2
   Khoản cùng 1 Điều sẽ expand ra 2 bản giống hệt nhau.

V5 sửa cả hai bằng cách xử lý NGAY Ở PHÍA B (không đợi generator sửa):
- Mỗi context CHỈ expand ra Điều nếu Điều đó <= PER_ITEM_BUDGET_SYLLABLES
  (ước lượng bằng .split(), khớp cách BTC đếm âm tiết) — nếu vượt, giữ
  nguyên Khoản gốc thay vì cắt ngang câu.
- Dedup theo text đã chuẩn hoá: nếu ứng viên tiếp theo trong ranked_units có
  text trùng (hoặc gần trùng, so sau khi chuẩn hoá whitespace) với 1 context
  đã chọn, BỎ QUA và lấy ứng viên xếp hạng kế tiếp — không giảm xuống dưới 3
  context nếu ranked_units còn đủ ứng viên khác.

Chạy: python pipeline/build_qa_packages_public_v5_top3_expand_budget.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config, io_utils
from src.b2_retrieval.retrieve import get_unit_text
from src.b6_context_package.package import build_context_item, build_qa_package, write_qa_packages_json

RERANK_PATH = Path("outputs/layer3/L3_rerank_public_aiteamvn.jsonl")  # MỚI — khác V5 đúng ở đây
OUT_PATH_JSON = config.OUTPUTS_DIR / "qa_packages_public_v5b_ranking_moi.json"
TOP_N = 3
# 300 âm tiết/context x 3 = 900 tối đa nếu cả 3 đều expand — nằm trong plateau
# 400-600..LEN_CAP=1200 đã dò trên dev_fast (baseline_template.py), có biên an
# toàn cho trường hợp cả 3 đều expand hết cỡ.
PER_ITEM_BUDGET_SYLLABLES = 300


def khoan_id_to_dieu_id(khoan_id: str) -> str:
    """{context_id}_{dieu_so}_{khoan_so} -> {context_id}_{dieu_so}."""
    parts = khoan_id.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else khoan_id


def infer_unit_type(unit_id: str, context_id: str) -> str:
    suffix = unit_id[len(context_id) + 1 :] if unit_id.startswith(context_id + "_") else ""
    n_parts = len([p for p in suffix.split("_") if p])
    if n_parts >= 2:
        return "khoan"
    if n_parts == 1:
        return "dieu_fallback"
    return "doc_fallback"


def syl_count(text: str) -> int:
    """Đếm âm tiết = .split(), KHỚP cách BTC chấm (xem baseline_template.py:syl)."""
    return len(text.split())


def normalize_for_dedup(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def decide_text(u: dict, parsed_corpus: dict) -> tuple[str, bool]:
    """Trả (text, da_expand). Chỉ expand Khoản->Điều nếu Điều vừa ngân sách;
    dieu_fallback/doc_fallback đã ở mức Điều/văn bản, không mở rộng thêm."""
    raw_text = u["text"]
    if u["unit_type"] != "khoan":
        return raw_text, False
    dieu_text = get_unit_text(parsed_corpus, khoan_id_to_dieu_id(u["unit_id"]))
    if dieu_text.strip() and syl_count(dieu_text) <= PER_ITEM_BUDGET_SYLLABLES:
        return dieu_text, True
    return raw_text, False


def main():
    print("Dang load parsed_corpus + public-official.json + ket qua Layer 3...", flush=True)
    parsed_corpus = io_utils.load_parsed_corpus()
    qa_public = io_utils.load_public_official()

    with open(config.OUTPUTS_DIR / "doc_number_index.json", encoding="utf-8") as f:
        doc_number_index = json.load(f)["usable"]
    context_id_to_doc_number = {v: k for k, v in doc_number_index.items()}

    n_empty_text = 0
    n_expanded = 0
    n_deduped_skips = 0
    n_underfilled = 0  # cau khong du 3 context duy nhat sau dedup
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

            # KHÔNG cắt top-N trước — duyệt hết ranked_units (đã sắp theo score),
            # bỏ qua ứng viên trùng/rỗng, dừng khi đủ TOP_N context DUY NHẤT.
            context_items = []
            seen_norm_texts: set[str] = set()
            for u in khoan_list:
                if len(context_items) >= TOP_N:
                    break
                text, expanded = decide_text(u, parsed_corpus)
                if not text.strip():
                    n_empty_text += 1
                    continue
                norm = normalize_for_dedup(text)
                if norm in seen_norm_texts:
                    n_deduped_skips += 1
                    continue
                seen_norm_texts.add(norm)
                if expanded:
                    n_expanded += 1
                doc = parsed_corpus.get(u["context_id"], {})
                context_items.append(build_context_item(
                    context_id=u["context_id"], unit_id=u["unit_id"], unit_type=u["unit_type"],
                    text=text, source_name=doc.get("name", ""), source_link=doc.get("link", ""),
                    document_number=context_id_to_doc_number.get(u["context_id"], ""),
                    retrieval_score=u["score"],
                ))

            if len(context_items) < TOP_N:
                n_underfilled += 1

            pkg = build_qa_package(qid, question, context_items, reference_answer)
            packages.append(pkg)

    n_written = write_qa_packages_json(packages, OUT_PATH_JSON)
    print(f"Da ghi {n_written} QAPackage -> {OUT_PATH_JSON}", flush=True)
    print(f"Kiem tra: so context rong (bi bo qua) = {n_empty_text}", flush=True)
    print(f"So context da expand Khoan->Dieu (trong ngan sach {PER_ITEM_BUDGET_SYLLABLES} am tiet) = {n_expanded}", flush=True)
    print(f"So ung vien bi bo qua vi TRUNG voi context da chon = {n_deduped_skips}", flush=True)
    print(f"So cau KHONG du {TOP_N} context duy nhat sau dedup = {n_underfilled} (ky vong nho, kiem tra neu > 5%)", flush=True)
    print(f"Kich thuoc file: {OUT_PATH_JSON.stat().st_size / 1024:.1f} KB", flush=True)


if __name__ == "__main__":
    main()
