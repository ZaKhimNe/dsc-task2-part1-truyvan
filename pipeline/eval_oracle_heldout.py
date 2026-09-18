"""
Đo TRẦN oracle của retrieval trên 198 câu heldout (train.json, nhãn B0
confidence>=0.6, unit_type="khoan") — trả lời câu hỏi: nếu B tìm ĐÚNG Khoản
100%, điểm METEOR tối đa có thể đạt là bao nhiêu, so với điểm THẬT hiện tại
(V1, retrieval của B tự tìm)?

KHÔNG cần corpus riêng của D (`selected-contexts.zip`) — B đã có đủ dữ liệu
(`data/parsed_corpus.jsonl`, `outputs/doc_number_index.json`). Tái dùng
`assemble()`/`cite_of()`/`topic_of()` từ `eval_harness/baseline_template.py`
(đã được D đo/hiệu chỉnh — không viết lại template mới), chỉ thay bước "tìm
đoạn trong văn bản" bằng việc lấy THẲNG text Khoản gold đã biết trước.

Đây là CẬN DƯỚI của oracle thật — 2 giới hạn cần nhớ:
  1. Lead/Conclusion ở đây là template cố định (D), không phải LLM thật (C)
     — C thật nhiều khả năng viết tốt hơn, nên oracle thật có thể cao hơn.
  2. Chỉ đo được trên 198/800 câu (198 câu có nhãn gold đủ tin) — không đại
     diện cho phân bố khó/dễ của toàn bộ 800 câu.

Output: 3 file trong outputs/oracle/ (gold_198.json, oracle_answers_198.json,
current_answers_198.json) — dùng eval_harness/evaluate.py chấm cả 2 phía sau đó:
    python evaluate.py --pred ../retrieval/outputs/oracle/oracle_answers_198.json --gold ../retrieval/outputs/oracle/gold_198.json
    python evaluate.py --pred ../retrieval/outputs/oracle/current_answers_198.json --gold ../retrieval/outputs/oracle/gold_198.json

Chạy: python pipeline/eval_oracle_heldout.py
"""
import json
import sys
from pathlib import Path

RETRIEVAL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RETRIEVAL_ROOT.parent
sys.path.insert(0, str(RETRIEVAL_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval_harness"))

from src.common import config, io_utils
from src.b2_retrieval.retrieve import get_unit_text
from baseline_template import assemble, cite_of, topic_of  # noqa: E402  (D's template, đã đo/hiệu chỉnh)

LABELS_PATH = config.OUTPUTS_DIR / "b0_labels_train_heldout.json"
CURRENT_ANSWERS_PATH = REPO_ROOT / "answers.json"  # V1 thật, đã có trong repo
OUT_DIR = config.OUTPUTS_DIR / "oracle"
CONFIDENCE_TRUST = config.B0_CONFIDENCE_TRUST


def khoan_id_parts(khoan_id: str) -> tuple[str, str, str]:
    """{context_id}_{dieu_so}_{khoan_so} -> (context_id, dieu_so, khoan_so)."""
    context_id, dieu_so, khoan_so = khoan_id.split("_", 2)
    return context_id, dieu_so, khoan_so


def main():
    labels_data = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    labels = labels_data["labels"]

    eligible: dict[str, dict] = {}
    for qid, spans in labels.items():
        khoans = [
            s for s in spans
            if s["unit_type"] == "khoan" and s["confidence"] >= CONFIDENCE_TRUST
        ]
        if khoans:
            eligible[qid] = khoans[0]  # span tin cậy nhất (đã lưu theo thứ tự tìm được)

    print(f"So cau du dieu kien (confidence>={CONFIDENCE_TRUST}, unit_type=khoan): {len(eligible)}")

    train = io_utils.load_train()
    parsed_corpus = io_utils.load_parsed_corpus()
    doc_number_index = json.loads((config.OUTPUTS_DIR / "doc_number_index.json").read_text(encoding="utf-8"))["usable"]
    context_id_to_doc_number = {v: k for k, v in doc_number_index.items()}
    current_answers_raw = json.loads(CURRENT_ANSWERS_PATH.read_text(encoding="utf-8"))

    gold_198: dict[str, dict] = {}
    oracle_answers: dict[str, dict] = {}
    current_answers: dict[str, dict] = {}
    n_missing_text = 0
    n_missing_current = 0

    for qid, span in eligible.items():
        if qid not in train:
            continue  # khong the xay ra (labels sinh tu chinh train.json), giu de an toan
        if qid not in current_answers_raw:
            n_missing_current += 1
            continue

        khoan_id = span["khoan_id"]
        context_id, dieu_so, khoan_so = khoan_id_parts(khoan_id)
        text = get_unit_text(parsed_corpus, khoan_id)
        if not text.strip():
            n_missing_text += 1
            continue

        question = train[qid]["question"]
        doc = parsed_corpus.get(context_id, {})
        # cite_of doc chi can "name" (+ "passage" neu muon regex tim so hieu ngay
        # trong noi dung — bo qua, doc_number_index cua B da dang tin hon).
        # Neu doc_number_index co so hieu, gan thang vao "name" de cite_of bat
        # duoc qua regex tren name (cach no dang lam voi title co san so hieu).
        doc_for_cite = {"name": doc.get("name", "")}
        cite = cite_of(doc_for_cite, f"Điều {dieu_so}.")
        answer = assemble(question, cite, [text])

        gold_198[qid] = train[qid]
        oracle_answers[qid] = {"answer": answer}
        current_answers[qid] = current_answers_raw[qid]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "gold_198.json").write_text(json.dumps(gold_198, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "oracle_answers_198.json").write_text(json.dumps(oracle_answers, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "current_answers_198.json").write_text(json.dumps(current_answers, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Da ghi {len(gold_198)} cau vao {OUT_DIR}/ (gold_198.json, oracle_answers_198.json, current_answers_198.json)")
    print(f"Bo qua: {n_missing_text} thieu text Khoan, {n_missing_current} thieu trong answers.json hien tai")


if __name__ == "__main__":
    main()
