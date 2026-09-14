"""
B6 — Đóng gói context cuối cùng, giao cho Generator (C).

Bài toán: gộp kết quả B1 (chunking) + B2 (retrieve) + B4 (span select)
thành 1 định dạng chuẩn hoá theo ĐÚNG schema Generator (C) đề xuất (nhận
13/08/2026), để Generator dùng làm input sinh câu trả lời. **Đây là điểm
dừng của B** — B không sinh câu trả lời cuối cùng.

REBUILD 13/08/2026 — đổi hẳn cấu trúc theo schema Generator đề xuất (khác
`ContextPackage` phẳng cũ, 1 record/1 context):
  {
    "id": "<question_id>",
    "question": "<câu hỏi>",
    "contexts": [{"document": {...}, "article": "Điều X Khoản Y", "text": "..."}],
    "reference_answer": "<answer thật, chỉ có ở train/warmup, None ở public-official>"
  }

Khác biệt chính so với bản cũ:
  - 1 record = 1 CÂU HỎI — `contexts` vẫn là LIST theo schema (tương thích
    nhiều context nếu Generator đổi ý sau này), nhưng THỰC TẾ đang ghi ĐÚNG
    1 phần tử. Lịch sử: 13/08 từng chốt trả TOP-3 Khoản (nguyên tắc ưu tiên
    recall "thà dư hơn thiếu") — ĐÃ ĐẢO NGƯỢC 31/08/2026: C xác nhận
    Generator COPY MÁY MÓC đoạn trích luật từ context (không LLM tự chọn
    giữa nhiều context), nên chỉ xử lý được ĐÚNG 1 context/câu hỏi. Xem
    `pipeline/build_qa_packages_heldout.py` (TOP_N=1) — bù lại việc mất top-
    2/3 dự phòng, context được MỞ RỘNG trả cả Điều (không chỉ Khoản), đo
    được Hit rate 46,5%→52,2% (xem `eval_context_expansion_heldout.py`).
  - Thêm `question` (text câu hỏi, trước đây chỉ có `question_id`) và
    `reference_answer` (answer thật, trước đây B6 không hề chạm vào answer
    — giờ Generator cần nó luôn trong package để train/eval, không phải tự
    join lại `train.json`/`warmup.json`).
  - `article`/`clause` là 2 FIELD RỜI ("Điều 4" / "khoản 26") — KHÔNG gộp
    thành 1 chuỗi như bản đầu. Xác nhận với C (31/08/2026): field
    `citation_metadata.articles`/`clauses` phía C cũng tách rời (xem
    `data123/data/train/baseline_eligible.json`), Generator muốn khớp
    convention đó. Vẫn suy từ `unit_id` bằng `_parse_unit_id`.
  - `document_number`: THÊM MỚI (31/08/2026, theo yêu cầu C) — số hiệu văn
    bản (vd "219/2013/TT-BTC"), tra từ `outputs/doc_number_index.json`
    (`build_doc_number_index.py`) qua `context_id`. "" nếu context không
    trích được số hiệu (~5.2% corpus) hoặc số hiệu bị trùng (ambiguous).
  - `document`: dict `{"name", "link"}` (đề xuất — Generator ghi "metadata
    hoặc parse từ corpus", còn mơ hồ, ĐANG DÙNG TẠM cấu trúc này, cần xác
    nhận lại). `retrieval_score` giữ thêm trong mỗi context item (Generator
    không yêu cầu nhưng không hại gì, hữu ích để họ tự lọc/trọng số nếu cần).

TRẠNG THÁI: theo schema Generator đề xuất, còn 1 điểm CHƯA xác nhận (cấu
trúc chính xác của `document`) — xem README.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypedDict


class ContextItem(TypedDict):
    document: dict  # {"name": str, "link": str} — ĐANG DÙNG TẠM, chưa xác nhận với Generator
    document_number: str  # "219/2013/TT-BTC" — "" nếu context không trích được số hiệu (~5.2% corpus, xem build_doc_number_index.py) hoặc số hiệu bị trùng (ambiguous)
    article: str  # "Điều X" — "" nếu doc_fallback (không có cấu trúc Điều)
    clause: str  # "khoản Y" — "" nếu dieu_fallback/doc_fallback (không có Khoản)
    text: str
    retrieval_score: float  # ngoài spec Generator, thêm để họ tự lọc/trọng số nếu cần


class QAPackage(TypedDict):
    id: str
    question: str
    contexts: list[ContextItem]
    reference_answer: str | None


def _parse_unit_id(unit_id: str, context_id: str) -> tuple[str, str]:
    """Suy dieu_so/khoan_so từ unit_id (dieu_id="{context_id}_{dieu_so}"
    hoặc khoan_id="{context_id}_{dieu_so}_{khoan_so}", xem b1_parser).
    unit_id không bắt đầu bằng "{context_id}_" (không đúng format kỳ vọng)
    -> trả ("", "") an toàn, không crash."""
    prefix = f"{context_id}_"
    if not unit_id.startswith(prefix):
        return "", ""
    suffix = unit_id[len(prefix) :]
    parts = suffix.split("_", 1)
    dieu_so = parts[0]
    khoan_so = parts[1] if len(parts) > 1 else ""
    return dieu_so, khoan_so


def _format_article(dieu_so: str) -> str:
    """"Điều X" — "" nếu không có cấu trúc Điều (doc_fallback)."""
    return f"Điều {dieu_so}" if dieu_so else ""


def _format_clause(khoan_so: str) -> str:
    """"khoản Y" — "" nếu không có Khoản (dieu_fallback/doc_fallback)."""
    return f"khoản {khoan_so}" if khoan_so else ""


def build_context_item(
    context_id: str,
    unit_id: str,
    unit_type: Literal["khoan", "dieu_fallback", "doc_fallback"],
    text: str,
    source_name: str,
    source_link: str,
    document_number: str,
    retrieval_score: float,
) -> ContextItem:
    """Đóng gói 1 context (1 Khoản/Dieu đã chọn) thành `ContextItem` — dùng
    cho từng phần tử trong `contexts` list của `build_qa_package`."""
    dieu_so, khoan_so = _parse_unit_id(unit_id, context_id)
    return ContextItem(
        document={"name": source_name, "link": source_link},
        document_number=document_number,
        article=_format_article(dieu_so),
        clause=_format_clause(khoan_so),
        text=text,
        retrieval_score=retrieval_score,
    )


def build_qa_package(
    question_id: str,
    question: str,
    context_items: list[ContextItem],
    reference_answer: str | None = None,
) -> QAPackage:
    """Đóng gói 1 câu hỏi + TOÀN BỘ context đã chọn (top-N, xem
    `selection.select_top_n_khoan`) thành `QAPackage` theo đúng schema
    Generator. `reference_answer=None` cho `public-official.json` (không có
    answer) — đúng như spec ("chỉ có ở train/dev")."""
    return QAPackage(
        id=question_id,
        question=question,
        contexts=context_items,
        reference_answer=reference_answer,
    )


def write_qa_packages_json(packages: list[QAPackage], json_path: Path) -> int:
    """Ghi list QAPackage ra 1 file `.json` duy nhất (Generator chỉ đọc `.json`,
    không cần bản `.jsonl` trung gian). Trả về số package đã ghi."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(packages, f, ensure_ascii=False, indent=2)
    return len(packages)
