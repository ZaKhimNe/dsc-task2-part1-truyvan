"""
B3 (docs/BAO_CAO_TONG_HOP_B.md mục 6 tầng 2) — Đo "trần recall của chunking":
với các câu hỏi đã có nhãn B0 đáng tin (confidence >= B0_CONFIDENCE_TRUST),
kiểm tra chunking Điều/Khoản của B1 có làm "vỡ" đáp án hay không, tức đáp án
có cần thông tin trải rộng hơn 1 Khoản (cần cả Điều) hay 1 Khoản đã đủ.

Vì sao không dùng thẳng `confidence` đã lưu trong MatchedSpan: B0 chỉ align
quote gốc với TỪNG unit riêng lẻ rồi CHỌN unit tốt nhất — confidence chỉ nói
"khớp với ĐÚNG Khoản đó tốt tới đâu", không nói liệu unit LỚN HƠN (cả Điều
chứa Khoản đó) có khớp TỐT HƠN không. Script này tái tạo lại quote_core
(giống hệt cách B0 làm — cùng hàm `_split_by_amend_marker`/`_extract_quote_core`),
align lại với CẢ 2: (a) đúng Khoản B0 đã chọn, (b) toàn bộ Điều chứa Khoản đó.

  score_dieu ~ score_khoan (chênh <= DIFF_THRESHOLD)  => Khoản hiện tại ĐỦ
  score_dieu > score_khoan rõ rệt                      => quote cần thông tin
                                                           ngoài Khoản đã chọn
                                                           (chunking Khoản làm vỡ)

Chỉ xét span có unit_type="khoan" (Điều CÓ chia Khoản) — case dieu_fallback/
doc_fallback đã là cả Điều/văn bản, không có "Khoản nhỏ hơn" để so.

Chạy: python pipeline/measure_chunking_ceiling.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config, io_utils
from src.b0_autolabel.label import _split_by_amend_marker, _extract_quote_core, align_span, MIN_QUOTE_WORDS

DIFF_THRESHOLD = 0.03  # chênh lệch coi là "có ý nghĩa" (không phải nhiễu đo)
SANITY_EPS = 0.02  # dung sai khi đối chiếu lại confidence đã lưu

LABEL_SOURCES = [
    ("warmup", config.OUTPUTS_DIR / "b0_labels_warmup.json", io_utils.load_warmup),
    ("train_sample", config.OUTPUTS_DIR / "b0_labels_train_sample.json", io_utils.load_train),
]


def find_khoan_and_dieu(parsed_corpus: dict, context_id: str, khoan_id: str):
    doc = parsed_corpus.get(context_id)
    if doc is None:
        return None, None
    for dieu in doc["dieu"]:
        for khoan in dieu["khoan"]:
            if khoan["khoan_id"] == khoan_id:
                return khoan, dieu
    return None, None


def main():
    print("Đang load parsed_corpus...", flush=True)
    parsed_corpus = io_utils.load_parsed_corpus()

    n_total = 0
    n_khoan_du = 0
    n_can_dieu = 0
    n_skip_notfound = 0
    n_sanity_mismatch = 0
    diffs = []
    examples_can_dieu = []

    for src_name, label_path, load_qa_fn in LABEL_SOURCES:
        with open(label_path, encoding="utf-8") as f:
            raw = json.load(f)
        labels = raw["labels"] if "labels" in raw and "sample_ids" in raw else raw
        qa_set = load_qa_fn()
        print(f"Đang xử lý {src_name}: {len(labels)} câu có nhãn...", flush=True)

        for qid, spans in labels.items():
            answer = qa_set.get(qid, {}).get("answer")
            if not answer:
                continue
            parts = _split_by_amend_marker(answer)
            for span in spans:
                if span["unit_type"] != "khoan" or span["confidence"] < config.B0_CONFIDENCE_TRUST:
                    continue
                khoan, dieu = find_khoan_and_dieu(parsed_corpus, span["context_id"], span["khoan_id"])
                if khoan is None:
                    n_skip_notfound += 1
                    continue

                query_text = None
                for part_text, role in parts:
                    qt = _extract_quote_core(part_text)
                    if len(qt.split()) < MIN_QUOTE_WORDS:
                        continue
                    if role == span["role"]:
                        query_text = qt
                        break
                if query_text is None:
                    n_skip_notfound += 1
                    continue

                _, score_khoan = align_span(query_text, khoan["text"])
                if abs(score_khoan - span["confidence"]) > SANITY_EPS:
                    n_sanity_mismatch += 1
                _, score_dieu = align_span(query_text, dieu["text"])

                diff = score_dieu - score_khoan
                diffs.append(diff)
                n_total += 1
                if diff > DIFF_THRESHOLD:
                    n_can_dieu += 1
                    if len(examples_can_dieu) < 5:
                        examples_can_dieu.append((qid, span["context_id"], span["khoan_id"], score_khoan, score_dieu))
                else:
                    n_khoan_du += 1

    print(f"\n=== Kết quả (n={n_total}, bỏ qua {n_skip_notfound} do không tìm lại được unit/quote) ===", flush=True)
    print(f"Sanity check (score_khoan vs confidence đã lưu, lệch > {SANITY_EPS}): "
          f"{n_sanity_mismatch}/{n_total} ({n_sanity_mismatch / n_total:.1%} nếu >0, kỳ vọng ~0)", flush=True)
    if n_total:
        print(f"Khoản hiện tại ĐỦ (chênh <= {DIFF_THRESHOLD}): {n_khoan_du}/{n_total} = {n_khoan_du / n_total:.1%}")
        print(f"Cần thông tin ngoài Khoản (cả Điều khớp tốt hơn rõ rệt): {n_can_dieu}/{n_total} = {n_can_dieu / n_total:.1%}")
        avg_diff = sum(diffs) / len(diffs)
        print(f"Chênh lệch trung bình (score_dieu - score_khoan): {avg_diff:.4f}")
        print("\nVí dụ case 'cần cả Điều' (tối đa 5):")
        for qid, cid, kid, sk, sd in examples_can_dieu:
            print(f"  qid={qid} context={cid} khoan={kid} score_khoan={sk:.3f} score_dieu={sd:.3f}")


if __name__ == "__main__":
    main()
